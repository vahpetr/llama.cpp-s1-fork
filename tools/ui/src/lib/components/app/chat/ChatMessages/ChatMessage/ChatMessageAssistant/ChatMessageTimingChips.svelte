<script lang="ts">
	// W2: always-visible timing chips of an assistant message (flow totals on an agentic root) and
	// the turn summary line; click -> the raw timings
	import { formatDuration, formatPct, formatTps, timingChips, turnSummary } from '$lib/utils';

	interface Props {
		message: DatabaseMessage;
	}

	let { message }: Props = $props();

	let expanded = $state(false);
	let chips = $derived(timingChips(message.timings));
	let turn = $derived(turnSummary(message.timings));
	const k = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 100_000 ? 0 : 1)}k` : `${n}`);
</script>

{#if chips}
	<div class="mt-1 text-xs text-muted-foreground tabular-nums" data-timing-chips>
		<button
			class="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-left hover:text-foreground"
			onclick={() => (expanded = !expanded)}
			title="Show the raw timings"
			type="button"
		>
			{#if chips.totalMs !== undefined}
				<span>⏱ {formatDuration(chips.totalMs)}</span>
			{/if}

			{#if chips.ttftMs !== undefined}
				<span>· TTFT {formatDuration(chips.ttftMs)}</span>
			{/if}

			{#if chips.prefill}
				<span>
					· ⇢ {k(chips.prefill.fresh)} fresh + {k(chips.prefill.cached)} cached ({formatPct(
						chips.prefill.cacheShare
					)}) {formatDuration(chips.prefill.ms)}
					{formatTps(chips.prefill.tps)}
				</span>
			{/if}

			{#if chips.decode}
				<span>
					· ⇠ {k(chips.decode.n)} tok {formatDuration(chips.decode.ms)}
					{formatTps(chips.decode.tps)}
				</span>
			{/if}

			{#if chips.mtp !== undefined}
				<span>· MTP {formatPct(chips.mtp)}</span>
			{/if}
		</button>

		{#if turn}
			<div data-turn-line>
				turn {formatDuration(turn.wallMs)} = model {formatDuration(turn.modelMs)} (prefill {formatDuration(
					turn.prefillMs
				)} + decode {formatDuration(turn.decodeMs)}) + tools {formatDuration(turn.toolsMs)} + other
				{formatDuration(turn.otherMs)} · {turn.llmCalls} LLM {turn.llmCalls === 1
					? 'call'
					: 'calls'}
				{#if turn.toolCalls > 0}· {turn.toolCalls} tools{/if}
			</div>
		{/if}

		{#if expanded}
			<pre
				class="mt-1 max-h-64 overflow-auto rounded bg-muted/40 p-2 text-[11px] whitespace-pre-wrap">{JSON.stringify(
					message.timings,
					null,
					1
				)}</pre>
		{/if}
	</div>
{/if}
