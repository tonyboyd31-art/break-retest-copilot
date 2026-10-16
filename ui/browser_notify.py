"""Best-effort browser (desktop) notifications via the Web Notification API.

Streamlit renders custom HTML in a sandboxed iframe; browsers may refuse
notification permission there. This is therefore EXPERIMENTAL: in-dashboard
alerts and toasts are the dependable channel.
"""

from __future__ import annotations

import json

import streamlit.components.v1 as components


def push_browser_notifications(alerts: list[dict]) -> None:
    payload = json.dumps([{"title": a["title"][:120], "body": a["message"][:240]} for a in alerts])
    components.html(
        f"""
<script>
(function() {{
  const items = {payload};
  if (!("Notification" in window)) return;
  const show = () => items.forEach(i => {{ try {{ new Notification(i.title, {{ body: i.body }}); }} catch (e) {{}} }});
  if (Notification.permission === "granted") show();
  else if (Notification.permission !== "denied") Notification.requestPermission().then(p => {{ if (p === "granted") show(); }});
}})();
</script>""",
        height=0,
    )
