/**
 * W2.2: the fork's view of the open conversation's context, shared by the ring (the single
 * context trigger by the send button) and its card: used / free (the server's last measurement +
 * estimates, or the live count), the derived reserve + pre-turn minimum, the ring level, the
 * "compact?" hint, the compaction state and the card rows.
 */

import { CONTEXT_RESERVE } from '$lib/constants';
import { chatStore, contextStatsStore, conversationsStore, settingsStore } from '$lib/stores';
import {
	compactHint,
	contextBudget,
	contextStatus,
	measureContextUsage,
	popoverRows,
	ringLevel,
	sessionLine
} from '$lib/utils';

export function useContextRing() {
	const conversation = $derived(conversationsStore.activeConversation);
	const messages = $derived(conversationsStore.activeMessages as DatabaseMessage[]);
	const cfg = $derived(conversationsStore.getContextConfig(conversation?.id));
	const nCtx = $derived(contextStatsStore.contextTotal);
	const used = $derived(
		Math.max(measureContextUsage(messages, nCtx).used, contextStatsStore.contextUsed)
	);
	const free = $derived(nCtx !== null ? Math.max(0, nCtx - used) : null);
	const usedPct = $derived(nCtx ? Math.min(100, (used / nCtx) * 100) : 0);
	const budget = $derived(
		nCtx
			? contextBudget({
					nCtx,
					summaryPrompt: cfg.summaryPrompt,
					summaryTokens: cfg.summaryTokens,
					used
				})
			: null
	);
	const hintPct = $derived(
		Number(settingsStore.config.compactHintPercent) || CONTEXT_RESERVE.DEFAULT_COMPACT_HINT_PERCENT
	);
	const compacting = $derived(
		chatStore.compaction !== null && chatStore.compaction.convId === conversation?.id
	);
	const busy = $derived(
		chatStore.isLoading || chatStore.isStreaming() || chatStore.compaction !== null
	);
	const canCompact = $derived(
		!busy && nCtx !== null && messages.some((m) => m.role === 'user' && !m.isSynthetic)
	);
	const level = $derived(ringLevel({ free, hintPct, preTurnMin: budget?.preTurnMin, usedPct }));
	const hint = $derived(canCompact && compactHint(usedPct, hintPct));
	const status = $derived(
		contextStatus({
			autoCompact: cfg.autoCompact,
			free,
			minReply: CONTEXT_RESERVE.MIN_REPLY_TOKENS,
			preTurnMin: budget?.preTurnMin,
			reserve: budget?.reserve
		})
	);
	const fromConv = $derived(
		conversation?.compactedFromConversationId
			? conversationsStore.conversations.find(
					(c) => c.id === conversation?.compactedFromConversationId
				)
			: undefined
	);
	const intoConv = $derived(
		conversation?.compactedIntoConversationId
			? conversationsStore.conversations.find(
					(c) => c.id === conversation?.compactedIntoConversationId
				)
			: undefined
	);

	function rows(now: number) {
		return popoverRows({
			autoCompact: cfg.autoCompact,
			breakdown: budget
				? {
						instruction: budget.instructionTokens,
						slack: budget.slackTokens,
						summary: budget.summaryTokens
					}
				: undefined,
			preTurnMin: budget?.preTurnMin,
			reserve: budget?.reserve,
			speed: sessionLine(messages, now)
		});
	}

	function compactNow() {
		if (!conversation || !canCompact) return;

		void chatStore.compactConversation(conversation.id);
	}

	function cancelCompaction() {
		if (conversation) void chatStore.stopGenerationForChat(conversation.id);
	}

	return {
		get busy() {
			return busy;
		},
		cancelCompaction,
		get canCompact() {
			return canCompact;
		},
		get compacting() {
			return compacting;
		},
		get compactionTokens() {
			return chatStore.compaction?.tokens ?? 0;
		},
		compactNow,
		get conversation() {
			return conversation;
		},
		get free() {
			return free;
		},
		get fromConv() {
			return fromConv;
		},
		get hasOverrides() {
			return !!conversation?.contextSettings;
		},
		get hint() {
			return hint;
		},
		get intoConv() {
			return intoConv;
		},
		get level() {
			return level;
		},
		get nCtx() {
			return nCtx;
		},
		rows,
		get status() {
			return status;
		},
		get used() {
			return used;
		},
		get usedPct() {
			return usedPct;
		}
	};
}
