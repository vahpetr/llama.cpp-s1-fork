<script lang="ts">
	import ContextGaugeDial from './ContextGaugeDial.svelte';
	import {
		gaugePopup,
		gaugePopupReset,
		gaugeTriggerClick,
		gaugeTriggerEnter,
		gaugeTriggerKeydown,
		gaugeTriggerLeave,
		gaugeTriggerPointerDown
	} from './gauge-popup.svelte';
	import ChatScreenContextSettingsDialog from '$lib/components/app/chat/ChatScreen/ChatScreenContextSettingsDialog.svelte';
	import { ColorLevel } from '$lib/enums';
	import { useContextGauge } from '$lib/hooks/use-context-gauge.svelte';
	import { useContextRing } from '$lib/hooks/use-context-ring.svelte';
	import { chatStore, conversationsStore, settingsStore } from '$lib/stores';
	import { legacyContextFields } from '$lib/utils';
	import { onDestroy, untrack } from 'svelte';
	import { toast } from 'svelte-sonner';

	const gauge = useContextGauge();
	// W2.2: the ring is THE context trigger; colour and percent from the fork's view (hint %,
	// pre-turn minimum), pulsing while a compaction runs
	const ring = useContextRing();
	const LEVELS = {
		critical: ColorLevel.CRITICAL,
		ok: ColorLevel.OK,
		warning: ColorLevel.WARNING
	} as const;
	const LEGACY_NOTICE_KEY = 'llama-ui.w12-context-settings-notice';

	// the card's pin and the settings dialog belong to the conversation they were opened in: a
	// switch resets them, and so does the ring's teardown (no card / dialog without its trigger)
	let cardFor: string | undefined = undefined;

	$effect(() => {
		const id = conversationsStore.activeConversation?.id;

		if (id !== cardFor) {
			cardFor = id;
			untrack(() => gaugePopupReset());
		}
	});

	onDestroy(() => gaugePopupReset());

	// W1.2: the W1/W1.1 reserve and "messages kept" settings are no longer read - say so once
	$effect(() => {
		const legacy = legacyContextFields(
			settingsStore.config,
			conversationsStore.activeConversation?.contextSettings
		);

		if (legacy.length === 0) return;

		try {
			if (localStorage.getItem(LEGACY_NOTICE_KEY)) return;

			localStorage.setItem(LEGACY_NOTICE_KEY, '1');
		} catch {
			return;
		}

		toast.info(
			`Context settings changed: the ${legacy.join(' and ')} setting is no longer used. The reserve is ` +
				'now derived from the summary length; the kept tail is set in tokens.'
		);
	});

	$effect(() => {
		const conv = conversationsStore.activeConversation;

		untrack(() => chatStore.processing.setActiveConversation(conv?.id ?? null));
	});

	$effect(() => {
		const conv = conversationsStore.activeConversation;
		const messages = conversationsStore.activeMessages as DatabaseMessage[];

		if (!conv) return;

		if (chatStore.isLoading || chatStore.isStreaming()) return;

		if (messages.length === 0) {
			untrack(() => chatStore.processing.setState(conv.id, null));

			return;
		}

		untrack(() => chatStore.processing.restoreFromMessages(messages, conv.id));
	});

	$effect(() => {
		gauge.startMonitoring();
	});
</script>

<div class="flex items-center gap-1">
	{#if ring.hint && !ring.compacting}
		<button
			class="text-xs text-amber-600 underline-offset-2 hover:underline dark:text-amber-400"
			data-compact-hint
			onclick={ring.compactNow}
			title="Summarize this conversation and continue in a new one (this one is kept)"
			type="button"
		>
			compact?
		</button>
	{/if}

	<div
		aria-expanded={gaugePopup.open}
		aria-label="Context usage"
		aria-pressed={gaugePopup.pinned}
		class={[
			'flex h-5 w-5 cursor-pointer items-center justify-center',
			ring.compacting && 'animate-pulse'
		]}
		data-context-gauge-trigger
		onclick={gaugeTriggerClick}
		onkeydown={gaugeTriggerKeydown}
		onpointerdown={gaugeTriggerPointerDown}
		onpointerenter={gaugeTriggerEnter}
		onpointerleave={gaugeTriggerLeave}
		role="button"
		tabindex="0"
	>
		<ContextGaugeDial
			level={ring.nCtx ? LEVELS[ring.level] : ColorLevel.NEUTRAL}
			percent={ring.nCtx ? Math.round(ring.usedPct) : null}
		/>
	</div>
</div>

{#if ring.conversation}
	{#key ring.conversation.id}
		<ChatScreenContextSettingsDialog
			bind:open={gaugePopup.settingsOpen}
			convId={ring.conversation.id}
		/>
	{/key}
{/if}
