<script lang="ts">
	// W2.1: one muted timing line per assistant message - "total · decode t/s · MTP %"; the details
	// (TTFT, prefill + cache share, turn = model + tools + other) on hover, click -> details + raw
	import { chipDetails, chipLine } from '$lib/utils';

	interface Props {
		message: DatabaseMessage;
	}

	let { message }: Props = $props();

	let expanded = $state(false);
	let line = $derived(chipLine(message.timings));
	let details = $derived(chipDetails(message.timings));
</script>

{#if line}
	<div class="mt-1 text-xs text-muted-foreground/70 tabular-nums" data-timing-chips>
		<button
			class="text-left hover:text-muted-foreground"
			onclick={() => (expanded = !expanded)}
			title={details.join('\n')}
			type="button"
		>
			{line}
		</button>

		{#if expanded}
			<div class="mt-1 grid gap-0.5" data-timing-details>
				{#each details as d (d)}
					<div>{d}</div>
				{/each}

				<pre
					class="mt-1 max-h-64 overflow-auto rounded bg-muted/40 p-2 text-[11px] whitespace-pre-wrap">{JSON.stringify(
						message.timings,
						null,
						1
					)}</pre>
			</div>
		{/if}
	</div>
{/if}
