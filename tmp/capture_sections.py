from playwright.sync_api import sync_playwright
sections = [
    ('hero', '#hero'),
    ('domains', '#domains'),
    ('live', '#live'),
    ('workflow', '#workflow'),
    ('evidence', '#evidence'),
    ('innovation', '#innovation'),
]
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1200}, device_scale_factor=1)
    page.goto('http://127.0.0.1:7860/', wait_until='networkidle')
    page.evaluate('window.scrollTo(0,0)')
    for name, sel in sections:
        page.locator(sel).scroll_into_view_if_needed(timeout=5000)
        page.wait_for_timeout(700)
        page.screenshot(path=f'/home/yclcg/Transshield_final/tmp/{name}.png')
    browser.close()
