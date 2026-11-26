/*
 * Where the Cloudflare Pages copy of the page finds the backend (DEPLOYMENT.md, step 8).
 *
 * TODO: set the backend's public address once it is decided, as a WebSocket URL ending in /ws/live:
 *   window.LIPSCRIBE_BACKEND = window.LIPSCRIBE_BACKEND || 'wss://<backend domain>/ws/live';
 * Use wss:// (not https://) and no port: Coolify's proxy maps the public address to the
 * container's port 1442. Until this is set, the Pages site says "LipScribe is not connected yet",
 * unless the link carries ?backend=wss://.../ws/live.
 *
 * Never put the access code here: this file is public. The code goes in the link (?code=...).
 *
 * When the backend serves the page itself (backend/server.py), it sets
 * window.LIPSCRIBE_BACKEND = 'self' right after this file, so that copy talks to its own server.
 */
window.LIPSCRIBE_BACKEND = window.LIPSCRIBE_BACKEND || '';
