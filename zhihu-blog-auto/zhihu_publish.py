#!/usr/bin/env python3
"""
zhihu_publish.py

基于 Playwright 的知乎专栏自动发文脚本，流程参考 publish_csdn.py：
1. 支持 posts 目录批量处理，解析 YAML Front Matter。
2. 复用/保存登录态（zhihu_state.json），首次运行等待手动登录。
3. 自动填充标题与正文，可选跳过发布阶段调试。
4. 记录发布结果到 publish_log.json，方便下次跳过已成功文章。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import frontmatter
from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

WRITE_URL = "https://zhuanlan.zhihu.com/write"
SCRIPT_DIR = Path(__file__).resolve().parent
POSTS_DIR = SCRIPT_DIR / "posts"
LOGS_DIR = SCRIPT_DIR / "logs"
HISTORY_DIR = SCRIPT_DIR / "history"
STATE_FILE = SCRIPT_DIR / "zhihu_state.json"
PUBLISH_LOG_FILE = LOGS_DIR / "publish_log.json"
DEFAULT_TOPICS = ["人工智能"]

LOGS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def read_markdown(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")
    return path.read_text(encoding="utf-8")


def _fallback_common_config() -> Dict[str, Any]:
    try:
        from src.utils.yaml_file_utils import read_common as _read_common  # type: ignore

        data = _read_common() or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    config_path = SCRIPT_DIR / "config" / "common.yaml"
    if config_path.exists():
        try:
            import yaml  # type: ignore

            with open(config_path, "r", encoding="utf-8") as cfg:
                data = yaml.safe_load(cfg) or {}
                if isinstance(data, dict):
                    return data
        except Exception:
            pass

    return {}


def convert_md_to_html(md_filename: str, include_footer: bool = True) -> str:
    """
    将 Markdown 文件转换为 HTML，移除 Front Matter，并（可选）追加 footer。
    """
    md_path = Path(md_filename).expanduser().resolve()
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    directory = md_path.parent if md_path.parent != Path("") else Path.cwd()
    footer_suffix = "_with_footer" if include_footer else ""
    html_path = directory / f"{md_path.stem}{footer_suffix}.html"

    if html_path.exists():
        return str(html_path)

    content = md_path.read_text(encoding="utf-8")
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2].strip()

    temp_md = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".md", delete=False)
    try:
        temp_md.write(content)
        temp_md_path = temp_md.name
    finally:
        temp_md.close()

    pandoc_command = [
        "pandoc",
        "--standalone",
        "-f",
        "markdown",
        "-t",
        "html5",
        "--no-highlight",
        temp_md_path,
        "-o",
        str(html_path),
    ]

    pandoc_css_path = SCRIPT_DIR / "pandoc.css"
    if pandoc_css_path.exists():
        pandoc_command[1:1] = ["--css", str(pandoc_css_path)]

    try:
        subprocess.run(pandoc_command, check=True)
    finally:
        try:
            os.unlink(temp_md_path)
        except FileNotFoundError:
            pass

    common_config = _fallback_common_config()
    include_footer = include_footer and common_config.get("include_footer", False)
    if include_footer:
        footer_path = SCRIPT_DIR / "config" / "footer.html"
        if footer_path.exists():
            with open(footer_path, "r", encoding="utf-8") as footer_file, open(
                html_path, "a", encoding="utf-8"
            ) as destination_file:
                destination_file.write(footer_file.read())

    return str(html_path)


def move_post_to_history(md_path: Path, html_path: Path) -> None:
    """将已发布文章及其 HTML 副本移动到 history 目录。"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    def _safe_move(source: Path) -> None:
        if not source.exists():
            return
        target = HISTORY_DIR / source.name
        if target.exists():
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            target = HISTORY_DIR / f"{source.stem}_{timestamp}{source.suffix}"
        shutil.move(str(source), str(target))
        print(f"已移动 {source.name} -> {target}")

    _safe_move(md_path)
    _safe_move(html_path)


def extract_metadata(md_text: str, source: Path) -> Tuple[str, List[str], str]:
    """解析 front matter，返回标题、话题列表、正文内容。"""
    post = frontmatter.loads(md_text)
    content = post.content.strip()

    if isinstance(post.metadata.get("tags"), str):
        tags = [tag.strip() for tag in post.metadata["tags"].split(",") if tag.strip()]
    else:
        tags = post.metadata.get("tags") or []

    title = post.metadata.get("title")
    if not title:
        for line in content.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break
    title = title or source.stem

    return title, tags, content


def load_publish_log() -> Dict[str, Dict[str, str]]:
    if not PUBLISH_LOG_FILE.exists():
        return {}
    try:
        data = json.loads(PUBLISH_LOG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    data.pop("__meta__", None)
    return data


def save_publish_log(log: Dict[str, Dict[str, str]]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"__meta__": {"updated_at": datetime.now().isoformat()}}
    payload.update(log)
    PUBLISH_LOG_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fill_title(page, title: str) -> bool:
    selectors = [
        'textarea[placeholder*="请输入标题"]',
        'textarea[placeholder*="输入标题"]',
        'textarea.PublishPanel-titleInput',
    ]
    for sel in selectors:
        el = page.query_selector(sel)
        if not el:
            continue
        try:
            el.click()
            el.fill("")
            el.type(title)
            print(f"已填充标题 (selector={sel})")
            return True
        except PlaywrightError as exc:
            print(f"填写标题失败 selector={sel}: {exc}")
    print("未找到标题输入框，请确认页面是否加载完成。")
    return False


def _copy_html_via_helper(context, html_path: Path) -> bool:
    helper_page = context.new_page()
    try:
        helper_page.goto(html_path.resolve().as_uri(), timeout=60000)
        helper_page.wait_for_load_state("load")
        body = helper_page.locator("body")
        body.wait_for(state="visible", timeout=5000)
        body.click()
        mod = "Meta" if sys.platform == "darwin" else "Control"
        helper_page.keyboard.press(f"{mod}+a")
        helper_page.keyboard.press(f"{mod}+c")
        time.sleep(0.3)
        return True
    except PlaywrightError as exc:
        print(f"复制 HTML 内容失败: {exc}")
        return False
    finally:
        helper_page.close()


def fill_editor_with_html(page, context, html_file: str) -> bool:
    """通过临时页面复制 HTML，再粘贴到 Draft.js 编辑器。"""
    html_path = Path(html_file)
    if not html_path.exists():
        print(f"HTML 文件不存在: {html_path}")
        return False

    if not _copy_html_via_helper(context, html_path):
        return False

    selectors = [
        "div.DraftEditor-editorContainer div.public-DraftEditor-content",
        "div.RichTextEditor div.public-DraftEditor-content",
        'div[contenteditable="true"] div.public-DraftEditor-content',
    ]
    mod = "Meta" if sys.platform == "darwin" else "Control"

    for sel in selectors:
        try:
            locator = page.locator(sel).first
            locator.wait_for(state="visible", timeout=15000)
            locator.click()
            page.keyboard.press(f"{mod}+a")
            page.keyboard.press("Backspace")
            page.keyboard.press(f"{mod}+v")
            time.sleep(0.5)
            print(f"已粘贴正文 (selector={sel})")
            return True
        except PlaywrightError as exc:
            print(f"向编辑器写入失败 selector={sel}: {exc}")

    print("未找到可编辑区域，可能尚未登录或编辑器未加载。")
    return False


def ensure_topics(page, topics: List[str]) -> None:
    """尝试在发布面板中补充话题，失败时仅打印提示。"""
    if not topics:
        topics = []
    topics = topics or DEFAULT_TOPICS

    topic_trigger_selectors = [
        'button:has-text("添加话题")',
        'div.PublishPanel-topicSelector',
        'button.css-1gtqxw0'
    ]
    topic_input_selectors = [
        'input[placeholder*="输入话题"]',
        'input.PublishTopicSelector-input',
        'input[placeholder*="搜索话题"]',
        'div.css-4cffwv input.Input'
    ]

    for trigger in topic_trigger_selectors:
        try:
            locator = page.locator(trigger).first
            locator.wait_for(state="visible", timeout=2000)
            locator.click()
            break
        except PlaywrightError:
            continue

    for topic in topics:
        added = False
        for inp in topic_input_selectors:
            try:
                input_locator = page.locator(inp).first
                input_locator.wait_for(state="visible", timeout=2000)
                input_locator.fill(topic)
                time.sleep(0.3)
                page.keyboard.press("Enter")
                added = True
                print(f"已尝试添加话题: {topic}")
                break
            except PlaywrightError:
                continue
        if not added:
            print(f"未能找到话题输入框，跳过话题: {topic}")


def expand_publish_settings_if_needed(page) -> None:
    """确保底部发布设置面板展开，以便访问封面/话题等元素。"""
    panel_selector = "div.css-13mrzb0"
    try:
        panel_locator = page.locator(panel_selector).first
        if panel_locator.count() == 0:
            return
        if panel_locator.is_visible():
            return
    except PlaywrightError:
        return

    toggle_selectors = [
        'button.css-9dyic7',
        'div.css-1ppjin3 button:has-text("发布设置")',
        'button:has-text("发布设置")'
    ]
    for sel in toggle_selectors:
        try:
            locator = page.locator(sel).first
            locator.wait_for(state="visible", timeout=2000)
            locator.click()
            time.sleep(0.2)
            if page.locator(panel_selector).first.is_visible():
                print("已展开发布设置面板")
                return
        except PlaywrightError:
            continue


def click_publish_buttons(page, topics: List[str]) -> bool:
    def robust_click(selector: str, desc: str, timeout: int = 15000, force: bool = False) -> bool:
        locator = page.locator(selector)
        if locator.count() == 0:
            return False
        target = locator.first
        for attempt in range(3):
            try:
                target.wait_for(state="visible", timeout=timeout)
                target.scroll_into_view_if_needed()
                target.wait_for(state="attached", timeout=timeout)
                try:
                    target.wait_for(state="enabled", timeout=timeout)
                except PlaywrightError:
                    pass
                aria_disabled = target.get_attribute("aria-disabled")
                disabled_attr = target.get_attribute("disabled")
                if aria_disabled == "true" or disabled_attr is not None:
                    time.sleep(0.3)
                    continue
                target.click(timeout=5000, force=force)
                print(f"已点击 {desc} (selector={selector})")
                return True
            except PlaywrightError as exc:
                if attempt == 2 and not force:
                    return robust_click(selector, desc, timeout, force=True)
                print(f"点击 {desc} 失败 (selector={selector}) attempt={attempt+1}: {exc}")
                time.sleep(0.5)
        return False

    publish_selectors = [
        'button.Button--primary.Button--blue:has-text("发布")',
        'button.css-d0uhtl:has-text("发布")',
        'button.JmYzaky7MEPMFcJDLNMG:has-text("发布")',
        'div.css-1ppjin3 button:has-text("发布")',
        'button.PublishPanel-triggerButton',
        'button:has-text("发布")',
    ]
    clicked = any(robust_click(sel, "发布按钮") for sel in publish_selectors)
    if not clicked:
        print("未能触发发布弹窗。")
        return False

    time.sleep(1)

    try:
        page.wait_for_url(lambda url: "zhuanlan.zhihu.com/p/" in url, timeout=60000)
        print(f"文章发布成功: {page.url}")
    except PlaywrightTimeoutError:
        print("等待发布成功跳转超时，请手动确认是否已发布。")
    return True


def ensure_editor_loaded(page, timeout: int) -> bool:
    selectors = [
        'textarea[placeholder*="请输入标题"]',
        "div.DraftEditor-editorContainer",
    ]
    for sel in selectors:
        try:
            page.wait_for_selector(sel, timeout=timeout)
            return True
        except PlaywrightTimeoutError:
            continue
    return False


def main():
    parser = argparse.ArgumentParser(description="知乎专栏自动发布脚本（参考 publish_csdn 流程）")
    parser.add_argument("--headless", default="false", choices=["true", "false"], help="是否无头模式，默认 false")
    parser.add_argument("--login-timeout", type=int, default=180, help="首次登录等待时间（秒），默认 180")
    parser.add_argument("--skip-publish", action="store_true", help="仅填充标题和正文，不点击发布")
    parser.add_argument("--cooldown", type=int, default=20, help="两篇文章之间的等待秒数，默认 30")
    args = parser.parse_args()

    if not POSTS_DIR.exists():
        print(f"未找到 posts 目录（{POSTS_DIR.resolve()}），请先准备 Markdown 文件。")
        sys.exit(2)

    files_to_process = sorted(POSTS_DIR.glob("*.md"))
    if not files_to_process:
        print("posts 目录下没有 .md 文件，退出。")
        sys.exit(0)

    publish_log = load_publish_log()

    headless = args.headless.lower() == "true"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        context_kwargs = {
            "viewport": {"width": 1440, "height": 900},
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "locale": "zh-CN",
            "timezone_id": "Asia/Shanghai",
        }

        if STATE_FILE.exists():
            print(f"加载登录态: {STATE_FILE}")
            context = browser.new_context(storage_state=str(STATE_FILE), **context_kwargs)
            page = context.new_page()
        else:
            context = browser.new_context(**context_kwargs)
            page = context.new_page()
            print("首次运行，请在打开的浏览器中完成知乎登录。")
            page.goto(WRITE_URL, timeout=60000)
            if ensure_editor_loaded(page, args.login_timeout * 1000):
                context.storage_state(path=str(STATE_FILE))
                print(f"已保存登录态到 {STATE_FILE}")
            else:
                print("在指定时间内未检测到编辑器，可能尚未登录，可手动登录后重新运行。")

        for idx, file_path in enumerate(files_to_process, start=1):
            if publish_log.get(file_path.name, {}).get("published"):
                print(f"跳过已发布: {file_path.name}")
                continue

            print(f"\n===== 处理 {idx}/{len(files_to_process)}: {file_path.name} =====")
            try:
                md_text = read_markdown(file_path)
                title, topics, _ = extract_metadata(md_text, file_path)
                html_path_str = convert_md_to_html(str(file_path))
                html_path = Path(html_path_str)
                topics = topics or DEFAULT_TOPICS
                print(f"使用标题: {title}")
                print(f"准备话题: {topics}")

                page.goto(WRITE_URL, timeout=60000)
                if not ensure_editor_loaded(page, 20000):
                    raise RuntimeError("编辑器未在预期时间内加载。")

                if not fill_title(page, title):
                    raise RuntimeError("未能填充标题。")

                if not fill_editor_with_html(page, context, str(html_path)):
                    raise RuntimeError("未能填充正文，请确认页面状态。")

                time.sleep(2)

                if args.skip_publish:
                    print("--skip-publish 已启用，仅填充内容完成。")
                    continue

                if click_publish_buttons(page, topics):
                    publish_log[file_path.name] = {
                        "published": True,
                        "title": title,
                        "topics": topics,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    save_publish_log(publish_log)
                    move_post_to_history(file_path, html_path)
                else:
                    raise RuntimeError("发布按钮流程未完成。")

                print(f"完成: {file_path.name}")
                if idx < len(files_to_process):
                    print(f"等待 {args.cooldown} 秒后继续...")
                    import random
                    time_detal = args.cooldown + random.randint(0, 10)
                    time.sleep(time_detal)

            except Exception as exc:
                print(f"处理 {file_path.name} 时出错: {exc}")
                page.screenshot(path="error_screenshot.png", full_page=True)
                publish_log[file_path.name] = {
                    "published": False,
                    "error": str(exc),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                save_publish_log(publish_log)
                    
        context.close()
        browser.close()


if __name__ == "__main__":
    main()