<script lang="ts">
	// W2: a tool call's chip - duration (running clock while it executes), args / result size,
	// status / error
	import type { AgenticSection, ChatMessageToolCallTiming } from '$lib/types';
	import { formatBytes, formatDuration, toolChip } from '$lib/utils';
	import { onDestroy } from 'svelte';

	interface Props {
		section: AgenticSection;
		record?: ChatMessageToolCallTiming;
		executingSince?: number | null;
	}

	let { executingSince = null, record, section }: Props = $props();

	let now = $state(Date.now());
	let timer: ReturnType<typeof setInterval> | null = null;

	$effect(() => {
		if (executingSince != null && !record) {
			timer ??= setInterval(() => (now = Date.now()), 200);
		} else if (timer) {
			clearInterval(timer);
			timer = null;
		}
	});

	onDestroy(() => {
		if (timer) clearInterval(timer);
	});

	let chip = $derived(
		toolChip({
			args: section.toolArgs,
			executingSince,
			now,
			record,
			result: section.toolResult
		})
	);
</script>

<div
	class="-mt-1 mb-1 flex flex-wrap gap-x-2 pl-1 text-xs text-muted-foreground tabular-nums"
	data-tool-chip
>
	{#if chip.running}
		<span>⏳ {formatDuration(chip.durationMs)}</span>
	{:else if chip.ok === false}
		<span class="text-red-500">✗ {formatDuration(chip.durationMs)}</span>
	{:else if chip.durationMs !== undefined}
		<span>✓ {formatDuration(chip.durationMs)}</span>
	{/if}

	<span>· args {formatBytes(chip.argsBytes)}</span>

	{#if chip.resultBytes !== undefined}
		<span>· result {formatBytes(chip.resultBytes)}</span>
	{/if}

	{#if chip.error}
		<span class="max-w-full truncate text-red-500" title={chip.error}>· {chip.error}</span>
	{/if}
</div>
