/* Context reserve + auto-compaction (W1) */
export const CONTEXT_RESERVE = {
	// ~3 characters per token for prose; tool results (JSON, logs, numbers, non-Latin text)
	// tokenize denser: 2.5. Every estimate carries a +10 % margin on top.
	CHARS_PER_TOKEN: 3,
	// W2.1: the meter suggests "compact?" from this fill (%)
	DEFAULT_COMPACT_HINT_PERCENT: 80,
	DEFAULT_SUMMARY_PROMPT:
		'Stop here and do not continue the task. The context window is nearly full: this conversation ' +
		'will continue in a fresh context that holds only the system prompt, the summary you write now ' +
		'and the last few messages. Write a continuation summary that lets you carry on without the ' +
		'earlier messages: the goal and every request of the user, decisions made and why, the ' +
		'current state of the work (files, commands, results, errors, what is done and what is not), ' +
		'the exact facts, names, paths and numbers that matter, and the next steps. Be complete but ' +
		'dense, at most {{MAX_TOKENS}} tokens. Write only the summary.',
	DEFAULT_SUMMARY_TOKENS: 8192,
	DEFAULT_TAIL_TOKENS: 16384,
	ESTIMATE_MARGIN: 1.1,
	// rule 3: bias on <|im_end|> for an unfinished-reply continuation; max tokens of one
	GUARD_BIAS: -2.0,
	GUARD_MAX_TOKENS: 16384,
	// continuations are joined verbatim (continue_final_message); the 24-char overlap strip is a
	// fallback for servers without assistant prefill only - off
	JOIN_OVERLAP_FALLBACK: false,
	// rules 2 / 3: continuations per turn
	MAX_CONTINUATIONS: 2,
	// rough token cost of an image / audio / video attachment
	MEDIA_TOKENS: 1024,
	// chat-template overhead per message (role markers, separators)
	MESSAGE_OVERHEAD_TOKENS: 8,
	// below twice this, a capped reply gets no thinking budget (the answer needs the room)
	MIN_ANSWER_TOKENS: 128,
	// a turn whose tail would have less room than this compacts before it is sent
	MIN_REPLY_TOKENS: 256,
	// a summary shorter than this is not accepted
	MIN_SUMMARY_CHARS: 80,
	// a summary request needs at least this much room; less = cut further back
	MIN_SUMMARY_TOKENS: 512,
	// rule 2: the note before a partial reply carried into the new chat
	PARTIAL_NOTE:
		'The last assistant message below was cut by the context limit. Continue it exactly where it stopped, without repeating anything.',
	// rule 1: reply room a user turn starts with at least (n_ctx / 4 on small contexts)
	PRE_TURN_REPLY_TOKENS: 32768,
	SLACK_MIN_TOKENS: 2048,
	SLACK_SHARE: 0.1,
	SUMMARY_HEADER: 'Summary of the earlier conversation (compacted from "{{NAME}}"):',
	// share of the context the copied tail may take in the new chat (cap on the tail setting)
	TAIL_CONTEXT_SHARE: 0.25,
	TOOL_CHARS_PER_TOKEN: 2.5,
	// rule 3: a ':' / dash / table / list ending counts only below this many tokens
	UNFINISHED_SHORT_TOKENS: 60
} as const;
