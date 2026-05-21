from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 2200}, device_scale_factor=1)
    page.goto('http://127.0.0.1:7860/', wait_until='networkidle')
    page.screenshot(path='/home/yclcg/Transshield_final/tmp/web_demo_full.png', full_page=True)
    browser.close()
