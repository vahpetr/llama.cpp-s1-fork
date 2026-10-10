<script lang="ts">
	// W2.2: the fork's part of the context card (nothing dropped vs v54): status, compaction in
	// progress, last request speed / cache / MTP, compute, the reserve with its breakdown, the
	// previous / next compacted chat, "Compact now"
	import { ArrowLeft, ArrowRight, Loader2 } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import type { useContextRing } from '$lib/hooks/use-context-ring.svelte';
	import { RouterService } from '$lib/services/router.service';
	import { formatTokenCount } from '$lib/utils';

	interface Props {
		ring: ReturnType<typeof useContextRing>;
	}

	let { ring }: Props = $props();

	let now = $state(Date.now());

	$effect(() => {
		const id = setInterval(() => (now = Date.now()), 30_000);

		return () => clearInterval(id);
	});

	let rows = $derived(ring.rows(now));
</script>

<div class="flex flex-col gap-2 text-xs" data-context-fork-rows>
	{#if ring.compacting}
		<div class="flex items-center gap-1.5 text-foreground" data-compacting>
			<Loader2 class="h-3 w-3 animate-spin" />
			Compacting… {formatTokenCount(ring.compactionTokens)} tokens ·

			<button
				class="underline-offset-2 hover:underline"
				onclick={ring.cancelCompaction}
				type="button"
			>
				Cancel
			</button>
		</div>
	{:else if ring.status}
		<div class="text-amber-600 dark:text-amber-400" data-context-status>{ring.status}</div>
	{/if}

	<dl class="grid gap-1">
		{#each rows as r (r.label)}
			<div class="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-2">
				<dt class="text-muted-foreground">{r.label}</dt>

				<dd class="min-w-0 tabular-nums break-words">
					{r.value}

					{#if r.sub}
						<div class="text-muted-foreground">{r.sub}</div>
					{/if}
				</dd>
			</div>
		{/each}
	</dl>

	<div class="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-border/40 pt-2">
		{#if ring.fromConv}
			{@const from = ring.fromConv}

			<button
				class="flex min-w-0 items-center gap-1 hover:underline"
				onclick={() => goto(RouterService.chat(from.id))}
				title="The conversation this one continues (kept as it was)"
				type="button"
			>
				<ArrowLeft class="h-3 w-3 shrink-0" />

				<span class="max-w-[7rem] truncate">{from.name}</span>
			</button>
		{/if}

		{#if ring.intoConv}
			{@const into = ring.intoConv}

			<button
				class="flex min-w-0 items-center gap-1 hover:underline"
				onclick={() => goto(RouterService.chat(into.id))}
				title="This conversation was compacted; open the continuation"
				type="button"
			>
				<span class="max-w-[7rem] truncate">{into.name}</span>

				<ArrowRight class="h-3 w-3 shrink-0" />
			</button>
		{/if}

		<button
			class="ml-auto hover:underline disabled:pointer-events-none disabled:opacity-50"
			data-compact-now
			disabled={!ring.canCompact}
			onclick={ring.compactNow}
			title="Summarize this conversation and continue in a new one (this one is kept)"
			type="button"
		>
			Compact now
		</button>
	</div>
</div>
