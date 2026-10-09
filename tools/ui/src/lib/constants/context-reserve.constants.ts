/* Context reserve + auto-compaction (W1) */
export const CONTEXT_RESERVE = {
	// ~3 characters per token: errs on the high side for code and English prose
	CHARS_PER_TOKEN: 3,
	DEFAULT_KEEP_MESSAGES: 4,
	DEFAULT_RESERVE_TOKENS: 6144,
	DEFAULT_SUMMARY_PROMPT:
		'Stop here and do not continue the task. The context window is nearly full: this conversation ' +
		'will continue in a fresh context that holds only the system prompt, the summary you write now ' +
		'and the last few messages. Write a continuation summary that lets you carry on without the ' +
		'earlier messages: the goal and every request of the user, decisions made and why, the ' +
		'current state of the work (files, commands, results, errors, what is done and what is not), ' +
		'the exact facts, names, paths and numbers that matter, and the next steps. Be complete but ' +
		'dense, at most {{MAX_TOKENS}} tokens. Write only the summary.',
	// rough token cost of an image / audio / video attachment
	MEDIA_TOKENS: 1024,
	// chat-template overhead per message (role markers, separators)
	MESSAGE_OVERHEAD_TOKENS: 8,
	// a turn whose tail would have less room than this compacts before it is sent
	MIN_REPLY_TOKENS: 256,
	// a summary request needs at least this much room; less = cut further back
	MIN_SUMMARY_TOKENS: 512,
	SUMMARY_HEADER: 'Summary of the earlier conversation (compacted from "{{NAME}}"):',
	// share of the context the copied tail may take in the new chat
	TAIL_CONTEXT_SHARE: 0.25
} as const;
