/*
 * Where the Cloudflare Pages copy of the page finds the backend (DEPLOYMENT.md, step 7).
 *
 * Set this to the backend's WebSocket address on Coolify, e.g.
 *   window.LIPSCRIBE_BACKEND = 'wss://lipscribe-api.ucu-dct.org/ws/live';
 * Leave it '' to use the pretend backend in mock-backend.js.
 *
 * Load order in index.html: config.js, then mock-backend.js, then app.js. When the backend serves
 * the page itself (backend/server.py), it sets window.LIPSCRIBE_BACKEND = 'self' after this file,
 * so the page talks to the server it came from. A ?backend=... in the address still wins over both.
 */
window.LIPSCRIBE_BACKEND = window.LIPSCRIBE_BACKEND || '';
