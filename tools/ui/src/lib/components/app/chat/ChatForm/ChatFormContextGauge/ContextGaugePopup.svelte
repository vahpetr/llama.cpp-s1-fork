<script lang="ts">
	import { colorLevelBgClass, colorLevelTextClass } from './context-gauge';
	import ContextGaugeDetails from './ContextGaugeDetails.svelte';
	import ContextGaugeForkSection from './ContextGaugeForkSection.svelte';
	import ContextGaugeLoadModel from './ContextGaugeLoadModel.svelte';
	import {
		gaugeCardEnter,
		gaugeCardLeave,
		gaugePopup,
		gaugePopupClose
	} from './gauge-popup.svelte';
	import { Pin, Settings2 } from '@lucide/svelte';
	import { ColorLevel } from '$lib/enums';
	import { useContextGauge } from '$lib/hooks/use-context-gauge.svelte';
	import { useContextRing } from '$lib/hooks/use-context-ring.svelte';
	import { formatParameters } from '$lib/utils/formatters';

	const gauge = useContextGauge();
	// W2.2: used / % / remaining and the colour from the fork's view (same as the ring)
	const ring = useContextRing();
	const level = $derived(
		{ critical: ColorLevel.CRITICAL, ok: ColorLevel.OK, warning: ColorLevel.WARNING }[ring.level]
	);
	const used = $derived(ring.nCtx !== null ? ring.used : gauge.contextUsed);
	const percent = $derived(ring.nCtx ? Math.round(ring.usedPct) : gauge.contextPercent);
	const remaining = $derived(ring.free ?? gauge.contextAvailable ?? 0);

	// The gauge hook wraps a processing state instance that only follows the
	// live stream while its own monitoring flag is set, so the card instance
	// starts monitoring like the dial does.
	$effect(() => {
		gauge.startMonitoring();
	});

	let cardEl = $state<HTMLElement | null>(null);

	// Any press outside the card and outside the dial closes the card.
	// Presses on the dial are excluded because the dial handles its own
	// toggle; the listener only exists while the card is open.
	$effect(() => {
		if (!gaugePopup.open) return;

		const onPointerDown = (event: PointerEvent) => {
			const target = event.target;

			if (!(target instanceof Node)) return;

			if (cardEl?.contains(target)) return;

			if (target instanceof Element && target.closest('[data-context-gauge-trigger]')) return;

			gaugePopupClose();
		};

		document.addEventListener('pointerdown', onPointerDown, true);

		return () => document.removeEventListener('pointerdown', onPointerDown, true);
	});

	const showProgressBar = $derived(
		gauge.contextTotal !== null &&
			gauge.contextTotal > 0 &&
			(gauge.activeModelId !== null || gauge.isActiveModelLoaded)
	);
</script>

{#if gaugePopup.open}
	<div
		bind:this={cardEl}
		class="absolute z-50 w-72 max-w-[calc(100vw-1rem)] -translate-x-1/2 rounded-lg border border-border/50 bg-popover p-3 text-sm text-popover-foreground shadow-lg ring-1 ring-foreground/10"
		onpointerenter={gaugeCardEnter}
		onpointerleave={gaugeCardLeave}
		role="status"
		style="left: {gaugePopup.centerX}px; bottom: {gaugePopup.bottom}px"
	>
		<div class="flex flex-col gap-2">
			<div class="flex items-center gap-2">
				<span class="font-medium">Context</span>

				<span class="text-muted-foreground">·</span>

				<span class="font-mono text-muted-foreground">
					{formatParameters(used)}
					/ {gauge.contextTotal !== null ? formatParameters(gauge.contextTotal) : '-'}
				</span>

				<span class="ml-auto flex items-center gap-1">
					{#if gaugePopup.pinned}
						<Pin class="h-3 w-3 text-muted-foreground" />
					{/if}

					<button
						aria-label="Compaction settings"
						class="relative rounded p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground"
						data-context-settings
						onclick={() => (gaugePopup.settingsOpen = true)}
						title={ring.hasOverrides
							? 'Compaction settings (this chat has overrides)'
							: 'Compaction settings'}
						type="button"
					>
						<Settings2 class="h-3.5 w-3.5" />

						{#if ring.hasOverrides}
							<span class="absolute top-0 right-0 h-1.5 w-1.5 rounded-full bg-primary"></span>
						{/if}
					</button>
				</span>
			</div>

			{#if gauge.activeModelId !== null && !gauge.isActiveModelLoaded}
				<ContextGaugeLoadModel
					isLoading={gauge.isActiveModelLoading}
					modelId={gauge.activeModelId}
					onLoad={gauge.loadModel}
				/>
			{:else if showProgressBar}
				<div class="h-1.5 w-full overflow-hidden rounded-full bg-muted">
					<div
						class="h-full rounded-full transition-all duration-300 {colorLevelBgClass(level)}"
						style="width: {percent}%"
					></div>
				</div>

				<div class="flex justify-between text-xs text-muted-foreground">
					<span>
						<span class={colorLevelTextClass(level)}>{percent}%</span> used
					</span>

					<span>
						{formatParameters(remaining)} remaining
					</span>
				</div>
			{:else}
				<div class="text-xs text-muted-foreground">No context info available</div>
			{/if}

			{#if ring.conversation}
				<ContextGaugeForkSection {ring} />
			{/if}

			{#if gauge.hasAnyUsage}
				<ContextGaugeDetails
					averageTokensPerSecond={gauge.averageTokensPerSecond}
					cumulativeCacheTotal={gauge.cumulativeCacheTotal}
					cumulativeOutput={gauge.cumulativeOutput}
					cumulativeRead={gauge.cumulativeRead}
					currentCache={gauge.currentCache}
					currentFresh={gauge.currentFresh}
					currentOutput={gauge.currentOutput}
					currentRead={gauge.currentRead}
					kvTotal={gauge.kvTotal}
					transientDetails={gauge.transientDetails}
				/>
			{/if}
		</div>
	</div>
{/if}
