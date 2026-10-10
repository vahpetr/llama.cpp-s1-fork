<script lang="ts">
	import ChatScreenContextSettingsDialog from './ChatScreenContextSettingsDialog.svelte';
	import { ArrowLeft, ArrowRight, Loader2, Pin, Settings2, X } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { CONTEXT_RESERVE } from '$lib/constants';
	import { RouterService } from '$lib/services/router.service';
	import { chatStore, contextStatsStore, conversationsStore, settingsStore } from '$lib/stores';
	import {
		compactHint,
		contextBudget,
		formatTokenCount,
		legacyContextFields,
		measureContextUsage,
		popoverRows,
		sessionLine
	} from '$lib/utils';
	import { toast } from 'svelte-sonner';

	// W2.1: the permanent surface is the thin meter with `used / n_ctx` (and, from the hint
	// threshold, a tint + "compact?"); everything else lives in ONE popover - on hover, a click pins
	// it so its controls are usable: last request speed, cache share, MTP, this chat's compute,
	// reserve + compaction threshold, the compaction links, "Compact now" and the settings gear.
	const LEGACY_NOTICE_KEY = 'llama-ui.w12-context-settings-notice';

	let conversation = $derived(conversationsStore.activeConversation);
	let messages = $derived(conversationsStore.activeMessages as DatabaseMessage[]);
	// this chat's overrides (W1.1), else the global settings
	let cfg = $derived(conversationsStore.getContextConfig(conversation?.id));
	let hasOverrides = $derived(!!conversation?.contextSettings);
	let showSettings = $state(false);
	let hovered = $state(false);
	let pinned = $state(false);
	let open = $derived(hovered || pinned);
	// the pin belongs to the conversation it was set in: a switch closes the popover
	let pinnedFor: string | undefined = undefined;

	$effect(() => {
		const id = conversation?.id;

		if (id !== pinnedFor) {
			pinnedFor = id;
			pinned = false;
			hovered = false;
		}
	});
	let nCtx = $derived(contextStatsStore.contextTotal);
	// the server's last measurement (+ an estimate of what followed), or the live count while a
	// request runs
	let used = $derived(
		Math.max(measureContextUsage(messages, nCtx).used, contextStatsStore.contextUsed)
	);
	let free = $derived(nCtx !== null ? Math.max(0, nCtx - used) : null);
	let clock = $state(Date.now());

	$effect(() => {
		if (!open) return;

		clock = Date.now();

		const id = setInterval(() => (clock = Date.now()), 30_000);

		return () => clearInterval(id);
	});

	let usedPct = $derived(nCtx ? Math.min(100, (used / nCtx) * 100) : 0);
	let budget = $derived(
		nCtx
			? contextBudget({
					nCtx,
					summaryPrompt: cfg.summaryPrompt,
					summaryTokens: cfg.summaryTokens,
					used
				})
			: null
	);
	let reservePct = $derived(nCtx && budget ? Math.min(100, (budget.reserve / nCtx) * 100) : 0);
	let hintPercent = $derived(
		Number(settingsStore.config.compactHintPercent) || CONTEXT_RESERVE.DEFAULT_COMPACT_HINT_PERCENT
	);
	let rows = $derived(
		open
			? popoverRows({
					autoCompact: cfg.autoCompact,
					preTurnMin: budget?.preTurnMin,
					reserve: budget?.reserve,
					speed: sessionLine(messages, clock)
				})
			: []
	);

	// W1.2: the W1/W1.1 reserve and "messages kept" settings are no longer read - say so once
	$effect(() => {
		const legacy = legacyContextFields(settingsStore.config, conversation?.contextSettings);

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

	let compacting = $derived(
		chatStore.compaction !== null && chatStore.compaction.convId === conversation?.id
	);
	let busy = $derived(
		chatStore.isLoading || chatStore.isStreaming() || chatStore.compaction !== null
	);
	let canCompact = $derived(
		!busy && nCtx !== null && messages.some((m) => m.role === 'user' && !m.isSynthetic)
	);
	let hint = $derived(canCompact && compactHint(usedPct, hintPercent));
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

	function compactNow() {
		if (!conversation || !canCompact) return;

		void chatStore.compactConversation(conversation.id);
	}
</script>

{#if conversation && (messages.length > 0 || fromConv || intoConv)}
	<div
		class="pointer-events-auto sticky top-2 z-10 mx-auto mb-2 flex w-full max-w-[48rem] justify-end pl-10 md:pl-0"
	>
		<div
			class="relative"
			data-context-bar
			onmouseenter={() => (hovered = true)}
			onmouseleave={() => (hovered = false)}
			role="group"
		>
			<div
				class="flex items-center gap-2 rounded px-1.5 py-0.5 text-xs text-muted-foreground backdrop-blur"
			>
				<button
					aria-expanded={open}
					aria-label="Context usage"
					class="flex items-center gap-2"
					data-context-meter
					onclick={() => (pinned = !pinned)}
					title={pinned ? 'Unpin the context details' : 'Pin the context details'}
					type="button"
				>
					<span
						aria-valuemax={nCtx ?? 0}
						aria-valuemin={0}
						aria-valuenow={used}
						class="relative block h-1 w-24 overflow-hidden rounded-full bg-muted sm:w-28"
						role="meter"
					>
						<span
							style:width="{usedPct}%"
							class={[
								'absolute inset-y-0 left-0',
								hint ? 'bg-amber-500' : 'bg-muted-foreground/50'
							]}
						></span>

						{#if reservePct > 0}
							<span style:width="{reservePct}%" class="absolute inset-y-0 right-0 bg-foreground/10"
							></span>
						{/if}
					</span>

					{#if nCtx !== null}
						<span class="tabular-nums">{formatTokenCount(used)} / {formatTokenCount(nCtx)}</span>
					{/if}
				</button>

				{#if compacting}
					<span class="flex items-center gap-1 text-foreground" data-compacting>
						<Loader2 class="h-3 w-3 animate-spin" />
						{formatTokenCount(chatStore.compaction?.tokens ?? 0)}

						<button
							aria-label="Cancel the compaction"
							class="hover:text-foreground"
							onclick={() => conversation && chatStore.stopGenerationForChat(conversation.id)}
							type="button"
						>
							<X class="h-3 w-3" />
						</button>
					</span>
				{:else if hint}
					<button
						class="text-amber-600 underline-offset-2 hover:underline dark:text-amber-400"
						data-compact-hint
						onclick={compactNow}
						title="Summarize this conversation and continue in a new one (this one is kept)"
						type="button"
					>
						compact?
					</button>
				{/if}
			</div>

			{#if open}
				<div
					class="absolute top-full right-0 z-20 mt-1 w-72 rounded-md border border-border/60 bg-popover p-3 text-xs text-popover-foreground shadow-md"
					data-context-popover
				>
					<div class="mb-2 flex items-center justify-between font-medium">
						<span>
							Context {used.toLocaleString()} / {(nCtx ?? 0).toLocaleString()}
							<span class="font-normal text-muted-foreground"
								>· {(free ?? 0).toLocaleString()} free</span
							>
						</span>

						<span class="flex items-center gap-1">
							{#if pinned}
								<Pin class="h-3 w-3 text-muted-foreground" />
							{/if}

							<button
								aria-label="Context settings for this chat"
								class={[
									'rounded p-0.5 hover:bg-muted',
									hasOverrides ? 'text-foreground' : 'text-muted-foreground'
								]}
								data-context-settings
								onclick={() => (showSettings = true)}
								title={hasOverrides
									? 'Compaction settings (this chat has overrides)'
									: 'Compaction settings'}
								type="button"
							>
								<Settings2 class="h-3.5 w-3.5" />
							</button>
						</span>
					</div>

					<dl class="grid gap-1">
						{#each rows as r (r.label)}
							<div class="grid grid-cols-[6.5rem_1fr] gap-2">
								<dt class="text-muted-foreground">{r.label}</dt>

								<dd class="tabular-nums">
									{r.value}

									{#if r.sub}
										<div class="text-muted-foreground">{r.sub}</div>
									{/if}
								</dd>
							</div>
						{/each}
					</dl>

					<div
						class="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-border/40 pt-2"
					>
						{#if fromConv}
							<button
								class="flex items-center gap-1 hover:underline"
								onclick={() => goto(RouterService.chat(fromConv.id))}
								type="button"
							>
								<ArrowLeft class="h-3 w-3" />

								<span class="max-w-[8rem] truncate">{fromConv.name}</span>
							</button>
						{/if}

						{#if intoConv}
							<button
								class="flex items-center gap-1 hover:underline"
								onclick={() => goto(RouterService.chat(intoConv.id))}
								type="button"
							>
								<span class="max-w-[8rem] truncate">{intoConv.name}</span>

								<ArrowRight class="h-3 w-3" />
							</button>
						{/if}

						<button
							class="ml-auto hover:underline disabled:pointer-events-none disabled:opacity-50"
							data-compact-now
							disabled={!canCompact}
							onclick={compactNow}
							type="button"
						>
							Compact now
						</button>
					</div>
				</div>
			{/if}
		</div>
	</div>

	<ChatScreenContextSettingsDialog bind:open={showSettings} convId={conversation.id} />
{/if}
