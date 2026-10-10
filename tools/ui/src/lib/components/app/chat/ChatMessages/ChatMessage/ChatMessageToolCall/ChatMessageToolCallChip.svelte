<script lang="ts">
	// W2.1: a muted "duration · size" fragment on a tool block's header (a running clock while the
	// call executes); the header gives it the space the title leaves, it truncates with an ellipsis; the error only when it failed
	import type { AgenticSection, ChatMessageToolCallTiming } from '$lib/types';
	import { formatBytes, toolChip, toolFragment } from '$lib/utils';
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
	let fragment = $derived(toolFragment(chip));
	let title = $derived(
		`args ${formatBytes(chip.argsBytes)}` +
			(chip.resultBytes !== undefined ? ` · result ${formatBytes(chip.resultBytes)}` : '') +
			(chip.error ? ` · ${chip.error}` : '')
	);
</script>

{#if fragment || chip.error}
	<span
		class="block max-w-[24ch] min-w-0 truncate text-xs whitespace-nowrap text-muted-foreground/70 tabular-nums"
		data-tool-chip
		{title}
	>
		{fragment}

		{#if chip.error}
			<span class="text-red-500/80">· {chip.error}</span>
		{/if}
	</span>
{/if}
