/**
 * Two-camera live console.
 *
 * Camera transport remains `/ws/camera/{id}/`; frames remain JSON-wrapped
 * base64 JPEGs drawn to canvas. Recognition panels retain 10-second polling.
 */

// The app is served under a deployment sub-path (e.g. /faceid) by a proxy that
// does NOT strip it, so a bare "/ws/camera/1/" resolves against the DOMAIN root
// - which on the shared host belongs to another project entirely. Templates
// interpolate {{ PREFIX }}; a .js file cannot, so the page hands it over on
// data-url-prefix and every socket URL is built here.
function airiPrefix() {
    // Guarded: this runs before DOM-ready in some paths, and the JS contract
    // tests drive the module with a stubbed `window` and no `document` at all.
    // No prefix is the correct answer in both cases - that is how it is served
    // at the domain root.
    if (typeof document === 'undefined' || !document.querySelector) return '';
    const el = document.querySelector('[data-url-prefix]');
    return (el && el.dataset && el.dataset.urlPrefix)
        ? el.dataset.urlPrefix.replace(/\/$/, '') : '';
}

function airiSocketUrl(path) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}${airiPrefix()}${path}`;
}

function airiHttpUrl(path) {
    return `${airiPrefix()}${path}`;
}
(function initialiseLiveConsole() {
    'use strict';

    const MAX_RECENT_FEED_NODES = 30;
    const RECONNECT_DELAY_MS = 3000;

    function createElement(tagName, className, text) {
        const node = document.createElement(tagName);
        if (className) node.className = className;
        if (text !== undefined && text !== null) node.textContent = String(text);
        return node;
    }

    function safeMediaSource(value) {
        if (typeof value !== 'string' || !value.trim()) return '';
        const source = value.trim();
        if (source.startsWith('/') && !source.startsWith('//')) return source;
        try {
            const parsed = new URL(source, window.location.origin);
            if (parsed.origin === window.location.origin && ['http:', 'https:'].includes(parsed.protocol)) {
                return parsed.href;
            }
        } catch (error) {
            return '';
        }
        return '';
    }

    function formatTime(value) {
        if (!value) return '—';
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return String(value);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    function formatConfidence(value) {
        const score = Number(value);
        if (!Number.isFinite(score)) return 'Not Available';
        return `${(score <= 1 ? score * 100 : score).toFixed(1)}%`;
    }

    function normalizeSnapshotPath(value) {
        if (typeof value !== 'string' || !value.trim()) return null;
        const source = value.trim();
        if (source.startsWith('/') || /^https?:\/\//i.test(source)) return source;
        // A bare stored path. The server now sends these as prefixed URLs
        // (see viewmodels.live_event); anything still arriving bare is built
        // under the deployment prefix, never off the domain root.
        return `${airiPrefix()}/media/${source.replace(/^\/+/, '')}`;
    }

    function normalizeAttendanceEvent(data) {
        if (!data || data.type !== 'event') return null;
        const transition = typeof data.transition === 'string' ? data.transition : '';
        const action = transition === 'CHECK_IN'
            ? 'IN'
            : transition === 'CHECK_OUT'
                ? 'OUT'
                : data.role === 'IN' || data.role === 'OUT' ? data.role : '—';
        return {
            name: typeof data.name === 'string' ? data.name : "Unknown Employee",
            department: typeof data.department === 'string' ? data.department : '',
            camera: typeof data.camera === 'string' ? data.camera : '',
            action,
            transition,
            score: data.score,
            time: data.ts || data.time || '',
            snapshot: normalizeSnapshotPath(data.snapshot),
        };
    }

    function appendEvidence(container, source, title, subtitle, altText, fallbackIcon) {
        const safeSource = safeMediaSource(source);
        if (!safeSource) {
            const placeholder = createElement('div', 'live-evidence-placeholder');
            placeholder.setAttribute('aria-label', 'No evidence available');
            const icon = createElement('i', `bi ${fallbackIcon}`);
            icon.setAttribute('aria-hidden', 'true');
            placeholder.append(icon);
            container.append(placeholder);
            return;
        }

        const button = createElement('button', 'live-evidence-button');
        button.type = 'button';
        button.dataset.airiEvidence = '';
        button.dataset.evidenceSrc = safeSource;
        button.dataset.evidenceTitle = title || 'Evidence';
        button.dataset.evidenceSubtitle = subtitle || '';
        const image = createElement('img');
        image.src = safeSource;
        image.alt = altText;
        image.loading = 'lazy';
        button.append(image);
        container.append(button);
    }

    function renderRecentEvents(container, entries) {
        if (!container) return;
        const fragment = document.createDocumentFragment();
        const recent = Array.isArray(entries) ? entries.slice(0, MAX_RECENT_FEED_NODES) : [];

        recent.forEach((entry) => {
            const card = createElement('article', 'live-event-card');
            const action = entry.action || entry.transition || '—';
            const subtitle = [entry.camera, action, entry.time].filter(Boolean).join(' · ');
            appendEvidence(
                card, entry.snapshot, entry.name || 'Recognition', subtitle,
                `Best-quality evidence for ${entry.name || 'recognition'}`, 'bi-person-bounding-box',
            );

            const body = createElement('div', 'live-event-card__body');
            const heading = createElement('div', 'd-flex justify-content-between gap-2');
            const identity = createElement('div');
            identity.append(
                createElement('strong', '', entry.name || "Unknown Employee"),
                createElement('small', '', entry.department || "No department assigned"),
            );
            const badgeClass = action === 'IN' ? 'status-chip status-chip--success' : 'status-chip status-chip--warning';
            heading.append(identity, createElement('span', badgeClass, action));

            const details = createElement('dl');
            [
                ['Confidence', formatConfidence(entry.score)],
                ['Camera', entry.camera || '—'],
                ['Time', entry.time || '—'],
            ].forEach(([label, value]) => {
                const item = createElement('div');
                item.append(createElement('dt', '', label), createElement('dd', '', value));
                details.append(item);
            });
            body.append(heading, details);
            card.append(body);
            fragment.append(card);
        });

        if (!recent.length) {
            const empty = createElement('div', 'live-empty-state');
            empty.dataset.feedEmpty = '';
            const icon = createElement('i', 'bi bi-person-check');
            icon.setAttribute('aria-hidden', 'true');
            empty.append(icon, createElement('p', 'mb-0', "No recognitions yet."));
            fragment.append(empty);
        }
        container.replaceChildren(fragment);
        const count = document.getElementById('recentEventCount');
        if (count) count.textContent = String(recent.length);
    }

    function renderUnknownActivity(container, entries) {
        if (!container) return;
        const fragment = document.createDocumentFragment();
        const recent = Array.isArray(entries) ? entries.slice(0, MAX_RECENT_FEED_NODES) : [];

        recent.forEach((entry) => {
            const card = createElement('article', 'live-unknown-card');
            const title = `Unknown #${entry.id ?? '?'}`;
            const camera = entry.camera || (entry.camera_id ? `Camera #${entry.camera_id}` : '—');
            const lastSeen = entry.last_seen_label || formatTime(entry.last_seen || entry.timestamp);
            appendEvidence(
                card, entry.snapshot, title, `${camera} · ${lastSeen}`,
                "Best-quality evidence for unknown person", 'bi-person-exclamation',
            );
            const body = createElement('div');
            body.append(
                createElement('strong', '', title),
                createElement('small', '', `${camera} · ${lastSeen}`),
                createElement('span', '', `${entry.attempt_count ?? 0} frames`),
            );
            card.append(body);
            fragment.append(card);
        });

        if (!recent.length) {
            const empty = createElement('div', 'live-empty-state');
            empty.dataset.feedEmpty = '';
            const icon = createElement('i', 'bi bi-shield-check');
            icon.setAttribute('aria-hidden', 'true');
            empty.append(icon, createElement('p', 'mb-0', "No unknown activity recorded."));
            fragment.append(empty);
        }
        container.replaceChildren(fragment);
    }

    class CameraStreamManager {
        constructor() {
            this.streams = new Map();
        }

        connect(card) {
            const cameraId = card.dataset.cameraId;
            const canvas = card.querySelector('[data-stream-canvas]');
            const streamEndpoint = canvas?.dataset.streamEndpoint;
            if (!cameraId || !canvas || !streamEndpoint || this.streams.has(cameraId)) return;

            const wsUrl = airiSocketUrl(`/ws/camera/${streamEndpoint}/`);
            const stream = {
                card, canvas, streamEndpoint, socket: null, reconnectTimer: null,
                closed: false, frameSequence: 0, framesSeen: 0, image: null,
            };
            this.streams.set(cameraId, stream);

            const connectSocket = () => {
                if (stream.closed || stream.image) return;
                const socket = new WebSocket(wsUrl);
                stream.socket = socket;

                socket.addEventListener('open', () => {
                    this.updateState(stream, 'unavailable', 'Waiting for frame');
                });
                socket.addEventListener('message', (event) => this.decodeFrame(stream, event.data));
                socket.addEventListener('error', () => this.updateState(stream, 'unavailable', 'Unavailable'));
                socket.addEventListener('close', () => {
                    if (stream.closed) return;
                    // Closed before a single frame arrived. That is not a
                    // flaky connection - it is a deployment where the socket
                    // cannot work at all: uvicorn without `websockets`
                    // installed serves the handshake as ordinary HTTP (which
                    // the login middleware then answers with a 303), and a
                    // reverse proxy that does not forward Upgrade behaves the
                    // same way. Reconnecting forever just repaints "Unavailable"
                    // every three seconds, which is exactly what the
                    // live page did on aiscan.airi.uz. MJPEG needs neither
                    // the library nor the proxy's cooperation.
                    if (stream.framesSeen === 0) {
                        this.useMjpeg(stream);
                        return;
                    }
                    this.updateState(stream, 'unavailable', 'Unavailable');
                    if (stream.reconnectTimer === null) {
                        stream.reconnectTimer = window.setTimeout(() => {
                            stream.reconnectTimer = null;
                            connectSocket();
                        }, RECONNECT_DELAY_MS);
                    }
                });
            };

            stream.connectSocket = connectSocket;
            connectSocket();
        }

        /** Fall back to the MJPEG endpoint, which is plain HTTP. */
        useMjpeg(stream) {
            if (stream.image || stream.closed) return;
            const image = new Image();
            stream.image = image;
            image.className = 'live-camera-frame__mjpeg';
            image.alt = stream.canvas.getAttribute('aria-label') || '';
            image.addEventListener('load', () => {
                stream.card.querySelector('[data-stream-placeholder]')?.setAttribute('hidden', '');
                const lastFrame = stream.card.querySelector('[data-camera-metric="last-frame"]');
                if (lastFrame) lastFrame.textContent = new Date().toLocaleTimeString();
                this.updateState(stream, 'online', 'Online');
            });
            // A camera that is not running 404s here exactly as the socket
            // closed 4004, so say so rather than showing a broken image.
            image.addEventListener('error', () => {
                stream.card.querySelector('[data-stream-placeholder]')?.removeAttribute('hidden');
                this.updateState(stream, 'offline', 'Unavailable');
            });
            stream.canvas.setAttribute('hidden', '');
            stream.canvas.after(image);
            image.src = airiHttpUrl(`/video/${stream.streamEndpoint}`);
        }

        decodeFrame(stream, rawData) {
            try {
                const data = JSON.parse(rawData);
                if (data.type !== 'frame') return;
                // Counted before the stale check: a stale notice proves the
                // socket transport works, and the frameless-close fallback to
                // MJPEG must not be taken on a camera that is merely quiet.
                stream.framesSeen += 1;
                const frameSequence = ++stream.frameSequence;
                if (data.stale === true) {
                    // Sent WITHOUT a frame: the server stops repainting a
                    // source that has gone quiet rather than re-encoding its
                    // last image at 10 fps, and the last frame drawn stays
                    // on the canvas under this label.
                    this.updateState(stream, 'unavailable', 'Stream stale');
                    return;
                }
                if (!data.data) return;
                const image = new Image();
                image.addEventListener('load', () => {
                    if (frameSequence !== stream.frameSequence) return;
                    const context = stream.canvas.getContext('2d');
                    if (stream.canvas.width !== image.width || stream.canvas.height !== image.height) {
                        stream.canvas.width = image.width;
                        stream.canvas.height = image.height;
                    }
                    context.drawImage(image, 0, 0);
                    stream.card.querySelector('[data-stream-placeholder]')?.setAttribute('hidden', '');
                    const fps = stream.card.querySelector('[data-camera-metric="camera-fps"]');
                    const lastFrame = stream.card.querySelector('[data-camera-metric="last-frame"]');
                    const latency = stream.card.querySelector('[data-camera-metric="latency"]');
                    if (fps && data.fps !== undefined) {
                        const frameFps = Number(data.fps);
                        fps.textContent = Number.isFinite(frameFps) && frameFps > 0
                            ? String(data.fps) : 'Not Available';
                    }
                    if (lastFrame) lastFrame.textContent = new Date().toLocaleTimeString();
                    if (latency && data.delay_ms !== undefined) latency.textContent = `${data.delay_ms} ms`;
                    this.updateState(stream, 'online', 'Online');
                }, { once: true });
                image.src = `data:image/jpeg;base64,${data.data}`;
            } catch (error) {
                console.warn(`Failed to decode frame from camera ${stream.streamEndpoint}`, error);
            }
        }

        updateState(stream, state, label) {
            stream.card.dataset.streamState = state;
            const status = stream.card.querySelector('.live-camera-card__header [data-stream-state]');
            if (!status) return;
            status.dataset.streamState = state;
            status.textContent = label;
            status.classList.toggle('status-chip--success', state === 'online');
            status.classList.toggle('status-chip--warning', state === 'unavailable');
            status.classList.toggle('status-chip--danger', state === 'offline');
        }

        disconnect(cameraId) {
            const stream = this.streams.get(cameraId);
            if (!stream) return;
            stream.closed = true;
            if (stream.reconnectTimer !== null) window.clearTimeout(stream.reconnectTimer);
            stream.socket?.close();
            if (stream.image) {
                // An MJPEG response never ends on its own: the server's
                // generator loops until the client goes away. Dropping the
                // element without clearing src leaves that request - and its
                // JPEG encode, ten times a second - running for the life of
                // the page, one more each time the operator hits reconnect.
                stream.image.src = '';
                stream.image.remove();
                stream.image = null;
                stream.canvas.removeAttribute('hidden');
            }
            this.streams.delete(cameraId);
        }

        reconnectAll() {
            const cards = Array.from(document.querySelectorAll('[data-camera-id]'));
            Array.from(this.streams.keys()).forEach((cameraId) => this.disconnect(cameraId));
            cards.forEach((card) => this.connect(card));
        }

        disconnectAll() {
            Array.from(this.streams.keys()).forEach((cameraId) => this.disconnect(cameraId));
        }
    }

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = { CameraStreamManager, normalizeAttendanceEvent };
    }
    if (typeof document === 'undefined') return;

    document.addEventListener('DOMContentLoaded', () => {
        const consoleElement = document.querySelector('[data-live-console]');
        if (!consoleElement) return;

        const eventFeed = consoleElement.querySelector('[data-recent-feed]');
        const unknownFeed = consoleElement.querySelector('[data-unknown-feed]');
        const eventsScript = document.getElementById('initial-recognition-events');
        const unknownScript = document.getElementById('initial-unknown-activity');
        let recentEvents = JSON.parse(eventsScript?.textContent || '[]');
        let unknownActivity = JSON.parse(unknownScript?.textContent || '[]');
        const streamManager = new CameraStreamManager();
        document.querySelectorAll('[data-camera-id]').forEach((card) => streamManager.connect(card));
        window.cameraStreamManager = streamManager;

        renderRecentEvents(eventFeed, recentEvents);
        renderUnknownActivity(unknownFeed, unknownActivity);

        const hydrateFromApi = (payload) => {
            if (Array.isArray(payload.events)) {
                recentEvents = payload.events.slice(0, MAX_RECENT_FEED_NODES);
                renderRecentEvents(eventFeed, recentEvents);
            }
            if (Array.isArray(payload.unknown_attempts)) {
                unknownActivity = payload.unknown_attempts.slice(0, MAX_RECENT_FEED_NODES);
                renderUnknownActivity(unknownFeed, unknownActivity);
            }
            if (!payload.stats) return;
            const mappings = [
                ['knownCount', payload.stats.recognized_today],
                ['checkedInCount', payload.stats.checked_in_today],
                ['checkedOutCount', payload.stats.checked_out_today],
                ['unknownCount', payload.stats.unknown_attempts],
                ['attendanceSummaryText', payload.stats.summary],
            ];
            mappings.forEach(([id, value]) => {
                const target = document.getElementById(id);
                if (target && value !== undefined && value !== null) target.textContent = String(value);
            });
        };

        const refreshLogs = async () => {
            try {
                const response = await fetch(consoleElement.dataset.recognitionLogsUrl, {
                    headers: { Accept: 'application/json' },
                });
                if (response.ok) hydrateFromApi(await response.json());
            } catch (error) {
                console.warn('Could not refresh live logs', error);
            }
        };

        document.addEventListener('click', (event) => {
            const button = event.target.closest('[data-action]');
            if (!button || !consoleElement.contains(button)) return;
            if (button.dataset.action === 'reconnect-streams') {
                streamManager.reconnectAll();
            } else if (button.dataset.action === 'refresh-live-feed') {
                refreshLogs();
            } else if (button.dataset.action === 'fullscreen') {
                const canvas = button.closest('[data-camera-id]')?.querySelector('[data-stream-canvas]');
                if (!canvas) return;
                if (document.fullscreenElement) document.exitFullscreen();
                else canvas.requestFullscreen().catch((error) => console.warn("Could not open full screen", error));
            }
        });

        document.addEventListener('fullscreenchange', () => {
            document.querySelectorAll('[data-action="fullscreen"] span').forEach((label) => {
                label.textContent = document.fullscreenElement ? 'Exit Fullscreen' : "Full Screen";
            });
        });

        let recognitionSocket = null;
        try {
            recognitionSocket = new WebSocket(airiSocketUrl('/ws/attendance/'));
            recognitionSocket.addEventListener('message', (event) => {
                try {
                    const data = JSON.parse(event.data);
                    const nextEvent = normalizeAttendanceEvent(data);
                    if (nextEvent) {
                        recentEvents = [nextEvent, ...recentEvents].slice(0, MAX_RECENT_FEED_NODES);
                        renderRecentEvents(eventFeed, recentEvents);
                    }
                } catch (error) {
                    console.warn('WebSocket record invalid', error);
                }
            });
        } catch (error) {
            console.warn('WebSocket unavailable', error);
        }

        refreshLogs();
        const pollingTimer = window.setInterval(refreshLogs, 10000);
        window.addEventListener('beforeunload', () => {
            window.clearInterval(pollingTimer);
            recognitionSocket?.close();
            streamManager.disconnectAll();
        }, { once: true });
    });
})();
