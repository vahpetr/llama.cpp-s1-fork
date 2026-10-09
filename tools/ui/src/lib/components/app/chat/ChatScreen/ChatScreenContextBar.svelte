<script lang="ts">
	import { ArrowLeft, ArrowRight, Loader2, Shrink, X } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { CONTEXT_RESERVE } from '$lib/constants';
	import { RouterService } from '$lib/services/router.service';
	import { chatStore, contextStatsStore, conversationsStore, settingsStore } from '$lib/stores';
	import { formatTokenCount, measureContextUsage, readContextReserveConfig } from '$lib/utils';

	// W1: context usage meter + "Compact now" for the open conversation

	let conversation = $derived(conversationsStore.activeConversation);
	let messages = $derived(conversationsStore.activeMessages as DatabaseMessage[]);
	let cfg = $derived(readContextReserveConfig(settingsStore.config));
	let nCtx = $derived(contextStatsStore.contextTotal);
	// the server's last measurement (+ an estimate of what followed), or the live count while a
	// request runs
	let used = $derived(
		Math.max(measureContextUsage(messages, nCtx).used, contextStatsStore.contextUsed)
	);
	let free = $derived(nCtx !== null ? Math.max(0, nCtx - used) : null);
	let usedPct = $derived(nCtx ? Math.min(100, (used / nCtx) * 100) : 0);
	let reservePct = $derived(
		nCtx && cfg.reserve > 0 ? Math.min(100, (cfg.reserve / nCtx) * 100) : 0
	);
	let atReserve = $derived(
		free !== null && cfg.reserve > 0 && free < cfg.reserve + CONTEXT_RESERVE.MIN_REPLY_TOKENS
	);
	let compacting = $derived(
		chatStore.compaction !== null && chatStore.compaction.convId === conversation?.id
	);
	let busy = $derived(
		chatStore.isLoading || chatStore.isStreaming() || chatStore.compaction !== null
	);
	let canCompact = $derived(
		!busy && nCtx !== null && messages.some((m) => m.role === 'user' && !m.isSynthetic)
	);
	let fromConv = $derived(
		conversation?.compactedFromConversationId
			? conversationsStore.conversations.find(
					(c) => c.id === conversation?.compactedFromConversationId
				)
			: undefined
	);
	let intoConv = $derived(
		conversation?.compactedIntoConversationId
			? conversationsStore.conversations.find(
					(c) => c.id === conversation?.compactedIntoConversationId
				)
			: undefined
	);
	let title = $derived(
		nCtx === null
			? 'Context size unknown'
			: `Context: ${used.toLocaleString()} of ${nCtx.toLocaleString()} tokens used, ` +
					`${(free ?? 0).toLocaleString()} free; reserve ${cfg.reserve.toLocaleString()} ` +
					`(kept for the summary${cfg.autoCompact ? ', auto-compaction on' : ''})`
	);

	function compactNow() {
		if (!conversation || !canCompact) return;

		void chatStore.compactConversation(conversation.id);
	}
</script>

{#if conversation && (messages.length > 0 || fromConv || intoConv)}
	<div
		class="pointer-events-auto sticky top-2 z-10 mx-auto mb-2 flex w-full max-w-[48rem] flex-wrap items-center gap-x-3 gap-y-1 rounded-md border border-border/40 bg-background/80 px-3 py-1 pl-10 text-xs text-muted-foreground backdrop-blur md:pl-3"
		data-context-bar
	>
		{#if fromConv}
			<button
				class="flex items-center gap-1 hover:text-foreground"
				onclick={() => goto(RouterService.chat(fromConv.id))}
				title="Open the conversation this one continues (kept as it was)"
				type="button"
			>
				<ArrowLeft class="h-3 w-3" />

				<span class="max-w-[10rem] truncate">{fromConv.name}</span>
			</button>
		{/if}

		<div class="flex items-center gap-2" {title}>
			<div
				aria-label="Context usage"
				aria-valuemax={nCtx ?? 0}
				aria-valuemin={0}
				aria-valuenow={used}
				class="relative h-1.5 w-24 overflow-hidden rounded-full bg-muted sm:w-32"
				role="meter"
			>
				<div
					style:width="{usedPct}%"
					class={[
						'absolute inset-y-0 left-0',
						atReserve ? 'bg-amber-500' : usedPct >= 80 ? 'bg-amber-400' : 'bg-green-500'
					]}
				></div>

				{#if reservePct > 0}
					<div
						style:width="{reservePct}%"
						class="absolute inset-y-0 right-0 border-l border-foreground/40 bg-foreground/15"
					></div>
				{/if}
			</div>

			{#if nCtx !== null}
				<span class="tabular-nums">
					{formatTokenCount(used)} / {formatTokenCount(nCtx)}
				</span>
			{/if}
		</div>

		{#if compacting}
			<span class="flex items-center gap-1 text-foreground">
				<Loader2 class="h-3 w-3 animate-spin" />
				Compacting… {formatTokenCount(chatStore.compaction?.tokens ?? 0)} tokens
			</span>

			<button
				class="flex items-center gap-1 hover:text-foreground"
				onclick={() => conversation && chatStore.stopGenerationForChat(conversation.id)}
				title="Cancel the compaction"
				type="button"
			>
				<X class="h-3 w-3" />
				Cancel
			</button>
		{:else if atReserve && !intoConv}
			<span class="text-amber-600 dark:text-amber-400">
				{cfg.autoCompact ? 'Reserve reached: the next message compacts' : 'Reserve reached'}
			</span>
		{/if}

		{#if intoConv}
			<button
				class="flex items-center gap-1 hover:text-foreground"
				onclick={() => goto(RouterService.chat(intoConv.id))}
				title="This conversation was compacted; open the continuation"
				type="button"
			>
				<span class="max-w-[10rem] truncate">{intoConv.name}</span>

				<ArrowRight class="h-3 w-3" />
			</button>
		{/if}

		{#if !compacting}
			<button
				class="ml-auto flex items-center gap-1 rounded px-1.5 py-0.5 hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
				disabled={!canCompact}
				onclick={compactNow}
				title="Summarize this conversation and continue in a new one with the summary and the last messages (this one is kept)"
				type="button"
			>
				<Shrink class="h-3 w-3" />
				Compact now
			</button>
		{/if}
	</div>
{/if}
