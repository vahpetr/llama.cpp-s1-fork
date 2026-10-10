// Shared state for the context gauge popup. The dial and the card live in
// different DOM subtrees, so open state and placement are coordinated here.
// centerX and bottom place the card just above the dial: both are measured
// once at open time, relative to the closest form ancestor; the dial and
// the card share that positioning frame, so the values stay exact for the
// whole time the card is open.
// Mouse pointers open on hover with a short grace delay to travel from
// dial to card. W2.2: a click / tap / Enter PINS the card (it stays open so its
// controls are usable) and a second one unpins and closes it; a pinned card
// ignores the hover close and closes on an outside press or a conversation switch.
import {
	CONTEXT_GAUGE_CARD_HALF_WIDTH_PX,
	CONTEXT_GAUGE_CLOSE_GRACE_MS,
	CONTEXT_GAUGE_DIAL_GAP_PX,
	CONTEXT_GAUGE_EDGE_MARGIN_PX
} from '$lib/constants';

let closeTimer: ReturnType<typeof setTimeout> | undefined;
let lastPointerType = '';

export const gaugePopup = $state({
	bottom: 0,
	centerX: 0,
	detailsOpen: false,
	open: false,
	pinned: false,
	/** W2.2: the compaction settings dialog (mounted with the ring, opened from the card's gear) */
	settingsOpen: false
});

function openFrom(trigger: HTMLElement): void {
	clearTimeout(closeTimer);
	const frame = trigger.closest('form');

	if (frame) {
		const frameRect = frame.getBoundingClientRect();
		const triggerRect = trigger.getBoundingClientRect();
		const centerX = triggerRect.left + triggerRect.width / 2 - frameRect.left;
		const min = CONTEXT_GAUGE_CARD_HALF_WIDTH_PX + CONTEXT_GAUGE_EDGE_MARGIN_PX;
		const max = frameRect.width - CONTEXT_GAUGE_CARD_HALF_WIDTH_PX - CONTEXT_GAUGE_EDGE_MARGIN_PX;

		gaugePopup.centerX = Math.min(Math.max(centerX, min), Math.max(min, max));
		gaugePopup.bottom = frameRect.bottom - triggerRect.top + CONTEXT_GAUGE_DIAL_GAP_PX;
	}

	gaugePopup.open = true;
}

/** Pin toggle: pin (and open) an unpinned card, unpin and close a pinned one. */
function togglePin(trigger: HTMLElement): void {
	if (gaugePopup.pinned) {
		gaugePopupClose();
	} else {
		openFrom(trigger);
		gaugePopup.pinned = true;
	}
}

export function gaugePopupClose(): void {
	clearTimeout(closeTimer);
	gaugePopup.open = false;
	gaugePopup.pinned = false;
}

/**
 * W2.2: full reset - card closed and unpinned, the settings dialog closed. On a conversation
 * switch and when the ring unmounts (the card and the dialog must not outlive their trigger or
 * reopen for another conversation).
 */
export function gaugePopupReset(): void {
	gaugePopupClose();
	gaugePopup.detailsOpen = false;
	gaugePopup.settingsOpen = false;
}

/**
 * W2.2: does a press outside the card close it? Not on the ring (it toggles itself), not inside
 * the card, and not while the settings dialog (portalled outside the card) is open.
 */
export function outsidePressDismisses(p: {
	insideCard: boolean;
	onTrigger: boolean;
	settingsOpen: boolean;
}): boolean {
	return !p.insideCard && !p.onTrigger && !p.settingsOpen;
}

export function gaugeTriggerPointerDown(event: PointerEvent): void {
	lastPointerType = event.pointerType;
}

export function gaugeTriggerClick(event: MouseEvent): void {
	void lastPointerType;
	togglePin(event.currentTarget as HTMLElement);
}

export function gaugeTriggerKeydown(event: KeyboardEvent): void {
	if (event.key !== 'Enter' && event.key !== ' ') return;

	event.preventDefault();
	togglePin(event.currentTarget as HTMLElement);
}

export function gaugeTriggerEnter(event: PointerEvent): void {
	if (event.pointerType !== 'mouse') return;

	openFrom(event.currentTarget as HTMLElement);
}

export function gaugeTriggerLeave(event: PointerEvent): void {
	if (event.pointerType !== 'mouse') return;

	scheduleClose();
}

export function gaugeCardEnter(event: PointerEvent): void {
	if (event.pointerType !== 'mouse') return;

	clearTimeout(closeTimer);
}

export function gaugeCardLeave(event: PointerEvent): void {
	if (event.pointerType !== 'mouse') return;

	scheduleClose();
}

function scheduleClose(): void {
	clearTimeout(closeTimer);

	if (gaugePopup.pinned) return;

	closeTimer = setTimeout(() => {
		if (!gaugePopup.pinned) gaugePopup.open = false;
	}, CONTEXT_GAUGE_CLOSE_GRACE_MS);
}
