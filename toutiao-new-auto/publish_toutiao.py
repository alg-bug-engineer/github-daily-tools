#!/usr/bin/env python3
"""
publish_toutiao.py

参考 `csdn-blog-auto` 与 `zhihu-blog-auto` 的批处理设计，
使用 Playwright 提升头条发布的稳定性，并校验内容粘贴结果以保证正确性。
"""

from __future__ import annotations
import time
import argparse
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import frontmatter
from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

try:
    import markdown  # type: ignore
except ImportError as exc:  # pragma: no cover - 运行时兜底
    print("缺少 markdown 库，请先执行 `pip install markdown` 后重试。")
    raise SystemExit(1) from exc


PUBLISH_URL = "https://mp.toutiao.com/profile_v4/graphic/publish"
SCRIPT_DIR = Path(__file__).resolve().parent
POSTS_DIR = SCRIPT_DIR / "posts"
LOGS_DIR = SCRIPT_DIR / "logs"
HISTORY_DIR = SCRIPT_DIR / "history"
STATE_FILE = SCRIPT_DIR / "toutiao_state.json"
PUBLISH_LOG_FILE = LOGS_DIR / "publish_log.json"

TITLE_SELECTOR = 'textarea[placeholder*="请输入文章标题"]'
EDITOR_SELECTOR = 'div.publish-editor div.ProseMirror'
NO_COVER_SELECTOR = 'label.byte-radio:has-text("无封面")'
PREVIEW_BUTTON_SELECTOR = 'button.publish-btn-last:has-text("预览并发布")'
CONFIRM_BUTTON_SELECTOR = 'button.publish-btn-last:has-text("确认发布")'


LOGS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


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
    payload = {"__meta__": {"updated_at": datetime.now().isoformat()}}
    payload.update(log)
    PUBLISH_LOG_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def move_post_to_history(md_path: Path) -> None:
    if not md_path.exists():
        return
    target = HISTORY_DIR / md_path.name
    if target.exists():
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        target = HISTORY_DIR / f"{md_path.stem}_{timestamp}{md_path.suffix}"
    shutil.move(str(md_path), str(target))
    print(f"已归档 {md_path.name} -> {target}")


def parse_markdown(md_path: Path) -> Tuple[str, str, str]:
    md_text = md_path.read_text(encoding="utf-8")
    post = frontmatter.loads(md_text)
    title = post.get("title")
    if not title:
        for line in post.content.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break
    title = re.sub(r'["\']', "", title or md_path.stem)
    html = markdown.markdown(
        post.content,
        extensions=["fenced_code", "tables", "toc", "codehilite"],
        output_format="html5",
    )
    plain = post.content.strip()
    return title, html, plain


def copy_html_via_helper(context, html: str) -> bool:
    wrapper = f"<html><head><meta charset='utf-8'></head><body>{html}</body></html>"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    try:
        tmp.write(wrapper)
        tmp_path = Path(tmp.name)
    finally:
        tmp.close()

    helper_page = context.new_page()
    try:
        helper_page.goto(tmp_path.resolve().as_uri(), wait_until="load", timeout=60000)
        body = helper_page.locator("body")
        body.wait_for(state="visible", timeout=5000)
        body.click()
        mod = "Meta" if sys.platform == "darwin" else "Control"
        body.press(f"{mod}+a")
        body.press(f"{mod}+c")
        helper_page.wait_for_timeout(300)
        return True
    except PlaywrightError as exc:
        print(f"复制 HTML 内容失败: {exc}")
        return False
    finally:
        helper_page.close()
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def verify_editor_content(page, expected_plain: str) -> None:
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", "", text or "")

    expected = _normalize(expected_plain)[:2000]
    actual = page.eval_on_selector(EDITOR_SELECTOR, "el => el.innerText || ''")
    actual_norm = _normalize(actual)
    if not actual_norm:
        raise RuntimeError("编辑器内容为空")
    if expected and len(actual_norm) < max(50, int(len(expected) * 0.6)):
        raise RuntimeError("粘贴后的正文长度异常，疑似粘贴失败")


def paste_html_into_editor(page, context, html: str, plain_text: str) -> None:
    if not copy_html_via_helper(context, html):
        raise RuntimeError("无法复制 HTML 内容到剪贴板")

    editor = page.locator(EDITOR_SELECTOR).first
    editor.wait_for(state="visible", timeout=15000)
    editor.click()
    mod = "Meta" if sys.platform == "darwin" else "Control"
    page.keyboard.press(f"{mod}+a")
    page.keyboard.press("Backspace")
    page.keyboard.press(f"{mod}+v")
    page.wait_for_timeout(800)
    verify_editor_content(page, plain_text)
    print("✓ 正文粘贴完成并通过校验")
    time.sleep(2)


def ensure_no_cover(page) -> None:
    try:
        locator = page.locator(NO_COVER_SELECTOR).first
        locator.scroll_into_view_if_needed()
        locator.click()
        print("✓ 已选择无封面")
        time.sleep(1)
    except PlaywrightError:
        print("⚠ 无法设置无封面，可忽略")


def click_publish_buttons(page) -> None:
    def _click(selector: str, desc: str, timeout: int = 15000) -> None:
        locator = page.locator(selector).first
        locator.wait_for(state="visible", timeout=timeout)
        locator.scroll_into_view_if_needed()
        locator.click()
        print(f"✓ {desc}")

    page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
    page.wait_for_timeout(1000)
    _click(PREVIEW_BUTTON_SELECTOR, "已点击预览并发布")
    page.wait_for_timeout(2000)
    _click(CONFIRM_BUTTON_SELECTOR, "已确认发布")
    page.wait_for_timeout(4000)


def ensure_login(page, context, login_timeout: int, headless: bool) -> None:
    page.goto(PUBLISH_URL, wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_selector(TITLE_SELECTOR, timeout=8000)
        print("✓ 检测到已登录")
        return
    except PlaywrightTimeoutError:
        if headless:
            raise RuntimeError("无头模式下未检测到登录态，请先在非无头模式登录一次。")
        print(f"⚠ 未检测到登录，请在 {login_timeout} 秒内完成登录...")
        try:
            page.wait_for_selector(TITLE_SELECTOR, timeout=login_timeout * 1000)
            print("✓ 登录成功")
            context.storage_state(path=str(STATE_FILE))
        except PlaywrightTimeoutError as exc:
            raise RuntimeError("登录超时，请重试") from exc


def fill_title(page, title: str) -> None:
    locator = page.locator(TITLE_SELECTOR).first
    locator.wait_for(state="visible", timeout=10000)
    locator.click()
    locator.fill("")
    locator.type(title, delay=20)
    print(f"✓ 标题填充完成: {title}")
    time.sleep(1)


def process_post(page, context, md_path: Path, publish_log: Dict[str, Dict[str, str]]) -> None:
    title, html, plain = parse_markdown(md_path)
    page.goto(PUBLISH_URL, wait_until="domcontentloaded", timeout=60000)
    fill_title(page, title)
    paste_html_into_editor(page, context, html, plain)
    ensure_no_cover(page)
    click_publish_buttons(page)
    publish_log[md_path.name] = {
        "published": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "title": title,
    }
    save_publish_log(publish_log)
    move_post_to_history(md_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量发布今日头条文章")
    parser.add_argument("--headless", choices=["true", "false"], default="false", help="是否启用无头模式，默认为 false")
    parser.add_argument("--login-timeout", type=int, default=120, help="首次运行等待手动登录的秒数")
    args = parser.parse_args()

    if not POSTS_DIR.exists():
        print(f"未找到 posts 目录: {POSTS_DIR}")
        raise SystemExit(2)

    files = sorted(POSTS_DIR.glob("*.md"))
    if not files:
        print("posts 目录下没有待发布的 Markdown 文件")
        raise SystemExit(0)

    publish_log = load_publish_log()
    headless = args.headless.lower() == "true"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=["--disable-blink-features=AutomationControlled"])
        context_kwargs = {
            "viewport": {"width": 1440, "height": 900},
            **({"storage_state": str(STATE_FILE)} if STATE_FILE.exists() else {}),
        }
        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        try:
            ensure_login(page, context, args.login_timeout, headless)
        except RuntimeError as exc:
            print(f"✗ {exc}")
            context.close()
            browser.close()
            raise SystemExit(1)

        # 统一保存一次最新登录态
        context.storage_state(path=str(STATE_FILE))

        for idx, md_path in enumerate(files, start=1):
            if publish_log.get(md_path.name, {}).get("published"):
                print(f"跳过已发布: {md_path.name}")
                continue

            print(f"\n===== 处理 {idx}/{len(files)}: {md_path.name} =====")
            try:
                process_post(page, context, md_path, publish_log)
            except Exception as exc:
                print(f"✗ 处理 {md_path.name} 失败: {exc}")
                publish_log[md_path.name] = {
                    "published": False,
                    "error": str(exc),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_publish_log(publish_log)
                continue
            time.sleep(10)

        context.close()
        browser.close()


if __name__ == "__main__":
    main()
