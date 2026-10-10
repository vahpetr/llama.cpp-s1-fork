/**
 * chatStore - Chat lifecycle, streaming and message operations
 *
 * Owns the active conversation's chat state: sending messages, streaming
 * responses, editing/regeneration flows and per-conversation processing
 * activity. Composes the stream manager, message flows, activity ledger and
 * processing snapshot; persists through conversationsStore.
 *
 * Uses ChatService for the API layer and conversationsStore for persistence.
 */

import { goto } from '$app/navigation';
import {
	CONTEXT_RESERVE,
	CWD_CLEARED_TEXT,
	MESSAGE_EXTRA_STORAGE_VERSION,
	SYSTEM_MESSAGE_PLACEHOLDER,
	TITLE_GENERATION
} from '$lib/constants';
import {
	ErrorDialogType,
	MessageRole,
	MessageType,
	ReasoningEffort,
	StreamConnectionState
} from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import { RouterService } from '$lib/services/router.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { agenticStore } from '$lib/stores/agentic/index.svelte';
import { chatActivityStore } from '$lib/stores/chat/activity.svelte';
import { draftMessagesStore } from '$lib/stores/chat/drafts.svelte';
import { type ChatFlowsHost, ChatMessageFlows } from '$lib/stores/chat/flows.svelte';
import { chatProcessingStore } from '$lib/stores/chat/processing.svelte';
import { type ChatStreamHost, ChatStreamManager } from '$lib/stores/chat/streams.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { mcpStore } from '$lib/stores/mcp/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type {
	ApiChatMessageData,
	ChatMessagePromptProgress,
	ChatMessageTimings,
	ChatStreamCallbacks,
	DatabaseConversation,
	DatabaseMessage,
	DatabaseMessageExtra,
	ErrorDialogState
} from '$lib/types';
import {
	buildSummaryInstruction,
	cappedThinkingBudget,
	compactedConversationName,
	dropTrailingEmptyAssistants,
	estimateMessageTokens,
	estimateTextTokens,
	filterByLeafNodeId,
	findMessageById,
	formatCwdMessage,
	getConversationModel,
	isAbortError,
	measureContextUsage,
	normalizeModelName,
	parseToolsSnapshot,
	planSummaryRequest,
	replyTokenCap,
	reserveHoldsSummary,
	resolveContextReserveConfig,
	selectTailMessages,
	shouldCompact,
	summaryRejection,
	tailTokenBudget
} from '$lib/utils';
import { SvelteMap } from 'svelte/reactivity';
import { toast } from 'svelte-sonner';

/** A running context compaction (W1): the summary request of one conversation. */
export interface CompactionState {
	convId: string;
	/** owner token: cleanup releases only the compaction it started */
	id: number;
	/** summary tokens generated so far */
	tokens: number;
}

export type CompactionResult = { convId: string } | { cancelled: true } | { error: string };

class ChatStore implements ChatStreamHost, ChatFlowsHost {
	chatReasoningStates = new SvelteMap<string, boolean>();
	chatStreamingStates = new SvelteMap<
		string,
		{ response: string; messageId: string; model?: string | null }
	>();
	/** the running context compaction (W1), null when none */
	compaction = $state<CompactionState | null>(null);
	currentResponse = $state('');
	errorDialogState = $state<ErrorDialogState | null>(null);
	// true while the active conversation has a local pipe (send, attach or resume-wait)
	isLoading = $derived(this.activity.isLocal(conversationsStore.activeConversation?.id ?? ''));
	// true while the active conversation streams reasoning content but no visible content yet
	isReasoning = $derived(
		this.chatReasoningStates.get(conversationsStore.activeConversation?.id ?? '') ?? false
	);
	pendingEditMessageId = $state<string | null>(null);
	// resumable stream connection state for the active conversation
	// streaming -> bytes flowing normally, resuming -> waiting on /v1/stream reconnect, lost -> unrecoverable
	streamConnectionState = $state<StreamConnectionState>(StreamConnectionState.STREAMING);
	private abortControllers = new SvelteMap<string, AbortController>();
	private addFilesHandler: ((files: File[]) => void) | null = $state(null);
	private compactionSeq = 0;
	// message flows: edit, regenerate, continue, delete
	private flows = new ChatMessageFlows(this);
	private isEditModeActive = $state(false);
	private pendingDraftFiles = $state<ChatUploadedFile[]>([]);
	private pendingDraftMessage = $state<string>('');
	/** Reactive: queued pending messages for non-agentic streaming */
	private pendingMessages = new SvelteMap<
		string,
		{ content: string; extras?: DatabaseMessageExtra[] }
	>();
	private preEncodeAbortController: AbortController | null = null;

	// server-side stream sessions: discovery, attach/replay, resume retry, remote sync
	private streams = new ChatStreamManager(this);

	/** Conv activity (local pipe / remote session), composed here. */
	get activity() {
		return chatActivityStore;
	}

	/** Processing state, composed here so consumers have a single chat scope. */
	get processing() {
		return chatProcessingStore;
	}

	/**
	 * Abort the current agentic flow signal without clearing loading state.
	 * Used by "Send immediately" to force the agentic loop to exit so that
	 * the pending steering message can be re-sent.
	 *
	 * Any tool calls captured mid-stream are dropped before the abort so the
	 * pending message (or a manual follow-up) does not re-send a half-received
	 * tool call with invalid JSON arguments to the server. Mirrors what the
	 * Stop button already does through stopGenerationForChat.
	 */
	async abortCurrentFlow(convId: string): Promise<void> {
		await this.savePartialResponseIfNeeded(convId);
		const c = this.abortControllers.get(convId);

		if (c) {
			c.abort();
			this.abortControllers.delete(convId);
		}
	}

	async addMessage(
		role: MessageRole,
		content: string,
		type: MessageType = MessageType.TEXT,
		parent: string = '-1',
		extras?: DatabaseMessageExtra[],
		isSynthetic?: boolean
	): Promise<DatabaseMessage> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) throw new Error('No active conversation');

		let parentId: string | null = null;

		if (parent === '-1') {
			const am = conversationsStore.activeMessages;

			if (am.length > 0) parentId = am[am.length - 1].id;
			else {
				const all = await conversationsStore.getConversationMessages(activeConv.id);
				const r = all.find((m) => m.parent === null && m.type === 'root');

				parentId = r ? r.id : await DatabaseService.createRootMessage(activeConv.id);
			}
		} else parentId = parent;

		const message = await DatabaseService.createMessageBranch(
			{
				children: [],
				content,
				convId: activeConv.id,
				extra: extras,
				isSynthetic,
				role,
				timestamp: Date.now(),
				toolCalls: '',
				type
			},
			parentId
		);

		conversationsStore.addMessageToActive(message);
		await conversationsStore.updateCurrentNode(message.id);
		conversationsStore.updateConversationTimestamp();

		return message;
	}
	async addSystemPrompt(): Promise<void> {
		let activeConv = conversationsStore.activeConversation;

		if (!activeConv) {
			await conversationsStore.createConversation();
			activeConv = conversationsStore.activeConversation;
		}

		if (!activeConv) return;

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const rootId = rootMessage
				? rootMessage.id
				: await DatabaseService.createRootMessage(activeConv.id);
			const existingSystemMessage = allMessages.find(
				(m) => m.role === MessageRole.SYSTEM && m.parent === rootId
			);

			if (existingSystemMessage) {
				this.pendingEditMessageId = existingSystemMessage.id;

				if (!conversationsStore.activeMessages.some((m) => m.id === existingSystemMessage.id))
					conversationsStore.activeMessages.unshift(existingSystemMessage);

				return;
			}

			const am = conversationsStore.activeMessages;
			const firstActiveMessage = am.find((m) => m.parent === rootId);
			const systemMessage = await DatabaseService.createSystemMessage(
				activeConv.id,
				SYSTEM_MESSAGE_PLACEHOLDER,
				rootId
			);

			if (firstActiveMessage) {
				await DatabaseService.updateMessage(firstActiveMessage.id, {
					parent: systemMessage.id
				});
				await DatabaseService.updateMessage(systemMessage.id, {
					children: [firstActiveMessage.id]
				});
				const updatedRootChildren = rootMessage
					? rootMessage.children.filter((id: string) => id !== firstActiveMessage.id)
					: [];

				await DatabaseService.updateMessage(rootId, {
					children: [
						...updatedRootChildren.filter((id: string) => id !== systemMessage.id),
						systemMessage.id
					]
				});
				const firstMsgIndex = conversationsStore.findMessageIndex(firstActiveMessage.id);

				if (firstMsgIndex !== -1)
					conversationsStore.updateMessageAtIndex(firstMsgIndex, {
						parent: systemMessage.id
					});
			}

			conversationsStore.activeMessages.unshift(systemMessage);
			this.pendingEditMessageId = systemMessage.id;
			conversationsStore.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to add system prompt:', error);
		}
	}
	cancelPreEncode(): void {
		if (this.preEncodeAbortController) {
			this.preEncodeAbortController.abort();
			this.preEncodeAbortController = null;
		}
	}

	/**
	 * Resets the loading, streaming and processing state for a conversation
	 * after a generation ends or errors. Shared by the flows' exit paths.
	 */
	cleanupStreaming(convId: string): void {
		this.setChatLoading(convId, false);
		this.clearChatStreaming(convId);
		this.processing.setState(convId, null);
	}
	clearChatStreaming(convId: string, messageId?: string): void {
		// session aware: a stale generation must not wipe a newer one's streaming state on the
		// same conversation, that would drop the frozen stop identity and stop the wrong session
		if (messageId !== undefined) {
			const cur = this.chatStreamingStates.get(convId);

			if (cur && cur.messageId !== messageId) return;
		}

		this.chatStreamingStates.delete(convId);

		if (convId === conversationsStore.activeConversation?.id) this.currentResponse = '';
	}
	clearEditMode(): void {
		this.isEditModeActive = false;
		this.addFilesHandler = null;
	}

	clearPendingEditMessageId(): void {
		this.pendingEditMessageId = null;
	}

	clearPendingMessage(convId: string): void {
		this.pendingMessages.delete(convId);
	}

	/** Reset per-view state when (re)mounting the empty chat screen. */
	clearUIState(): void {
		this.currentResponse = '';
	}

	/**
	 * W1: compact a conversation whose context is (nearly) full. A summary request reuses the cached
	 * prompt where the server can (the conversation exactly as the chat sends it + an instruction
	 * message; when the context is full or has no room for the summary, cut before the last reply
	 * at a measured point so the server can rewind to a checkpoint - best effort: without a usable
	 * checkpoint the server re-reads that prefix). Then a new conversation = system prompt + the
	 * summary (an assistant note) + the last messages, linked to this one, which is kept. Opens
	 * the new conversation.
	 */
	async compactConversation(
		convId: string,
		opts: { auto?: boolean } = {}
	): Promise<CompactionResult> {
		// ownership is claimed before the first await: no second compaction, send or agentic flow
		// can start on this conversation in between, and cleanup releases only what this call owns
		if (this.compaction) return { error: 'A compaction is already running' };

		if (this.isChatLoadingInternal(convId)) return { error: 'The conversation is busy' };

		const owner: CompactionState = { convId, id: ++this.compactionSeq, tokens: 0 };

		this.compaction = owner;
		this.setChatLoading(convId, true);

		const abortController = this.getOrCreateAbortController(convId);
		const fail = (error: string): CompactionResult => {
			toast.error(`Compaction failed: ${error}`);

			return { error };
		};

		let result: CompactionResult = { error: 'not started' };

		try {
			result = await this.runCompaction(convId, owner, abortController, fail, opts);

			return result;
		} catch (error) {
			result =
				isAbortError(error) || abortController.signal.aborted
					? { cancelled: true }
					: fail(error instanceof Error ? error.message : String(error));

			return result;
		} finally {
			if (this.compaction?.id === owner.id) this.compaction = null;

			this.abortRequest(convId);
			this.setChatLoading(convId, false);
			this.processing.setState(convId, null);
			this.settleQueuedAfterCompaction(convId, result, !!opts.auto);
		}
	}

	consumePendingDraft(): { message: string; files: ChatUploadedFile[] } | null {
		if (!this.pendingDraftMessage && this.pendingDraftFiles.length === 0) return null;

		const d = { files: [...this.pendingDraftFiles], message: this.pendingDraftMessage };

		this.pendingDraftMessage = '';
		this.pendingDraftFiles = [];

		return d;
	}

	consumePendingMessage(
		convId: string
	): { content: string; extras?: DatabaseMessageExtra[] } | null {
		const msg = this.pendingMessages.get(convId);

		if (!msg) return null;

		this.pendingMessages.delete(convId);

		return msg;
	}

	/** n_ctx of the model a request goes to (per slot, from /props), null when unknown */
	contextSizeFor(model?: string | null): number | null {
		const id = model || modelsStore.activeModelId;
		const nCtx = id ? modelsStore.props.getModelContextSize(id) : null;

		// single-model mode: the server's own /props
		return nCtx ?? (serverStore.isRouterMode ? null : serverStore.contextSize);
	}

	async continueAssistantMessage(messageId: string): Promise<void> {
		return this.flows.continueAssistantMessage(messageId);
	}

	async createAssistantMessage(parentId?: string): Promise<DatabaseMessage> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) throw new Error('No active conversation');

		return await DatabaseService.createMessageBranch(
			{
				children: [],
				content: '',
				convId: activeConv.id,
				model: null,
				role: MessageRole.ASSISTANT,
				timestamp: Date.now(),
				toolCalls: '',
				type: MessageType.TEXT
			},
			parentId || null
		);
	}

	async deleteMessage(messageId: string): Promise<void> {
		return this.flows.deleteMessage(messageId);
	}

	/**
	 * Server-side stream sessions (discovery, attach/replay, resume retry,
	 * remote-running snapshot) live in ChatStreamManager.
	 */
	async discoverActiveStream(convId: string): Promise<void> {
		return this.streams.discoverActiveStream(convId);
	}

	dismissErrorDialog(): void {
		this.errorDialogState = null;
	}

	async editAssistantMessage(
		messageId: string,
		newContent: string,
		shouldBranch: boolean
	): Promise<void> {
		return this.flows.editAssistantMessage(messageId, newContent, shouldBranch);
	}

	async editMessageWithBranching(
		messageId: string,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	): Promise<void> {
		return this.flows.editMessageWithBranching(messageId, newContent, newExtras);
	}

	async editUserMessagePreserveResponses(
		messageId: string,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	): Promise<void> {
		return this.flows.editUserMessagePreserveResponses(messageId, newContent, newExtras);
	}

	getAddFilesHandler(): ((files: File[]) => void) | null {
		return this.addFilesHandler;
	}

	/** Convs with any activity (local pipe or remote session), sidebar spinners. */
	getAllLoadingChats(): string[] {
		return this.activity.loadingConvs;
	}

	/**
	 * Request options of a real turn. Conversation-specific options (reasoning effort) come
	 * from `conversation` when given, else from the active conversation.
	 */
	getApiOptions(conversation?: DatabaseConversation | null): Record<string, unknown> {
		const currentConfig = settingsStore.config;
		const hasValue = (value: unknown): boolean =>
			value !== undefined && value !== null && value !== '';
		const apiOptions: Record<string, unknown> = { stream: true, timings_per_token: true };

		if (serverStore.isRouterMode) {
			const modelName = modelsStore.selectedModelName;

			if (modelName) apiOptions.model = modelName;
		}

		if (currentConfig.systemMessage) apiOptions.systemMessage = currentConfig.systemMessage;

		if (currentConfig.disableReasoningParsing) apiOptions.disableReasoningParsing = true;

		if (currentConfig.excludeReasoningFromContext) apiOptions.excludeReasoningFromContext = true;

		// an explicit reasoning choice overrides the server default, DEFAULT sends nothing
		// (undefined = the active conversation)
		const effort = conversationsStore.preferences.getReasoningEffort(conversation);

		if (effort !== ReasoningEffort.DEFAULT) {
			apiOptions.enableThinking = effort !== ReasoningEffort.OFF;

			if (effort !== ReasoningEffort.OFF) apiOptions.reasoningEffort = effort;
		}

		if (hasValue(currentConfig.temperature))
			apiOptions.temperature = Number(currentConfig.temperature);

		if (hasValue(currentConfig.max_tokens))
			apiOptions.max_tokens = Number(currentConfig.max_tokens);

		if (hasValue(currentConfig.dynatemp_range))
			apiOptions.dynatemp_range = Number(currentConfig.dynatemp_range);

		if (hasValue(currentConfig.dynatemp_exponent))
			apiOptions.dynatemp_exponent = Number(currentConfig.dynatemp_exponent);

		if (hasValue(currentConfig.top_k)) apiOptions.top_k = Number(currentConfig.top_k);

		if (hasValue(currentConfig.top_p)) apiOptions.top_p = Number(currentConfig.top_p);

		if (hasValue(currentConfig.min_p)) apiOptions.min_p = Number(currentConfig.min_p);

		if (hasValue(currentConfig.xtc_probability))
			apiOptions.xtc_probability = Number(currentConfig.xtc_probability);

		if (hasValue(currentConfig.xtc_threshold))
			apiOptions.xtc_threshold = Number(currentConfig.xtc_threshold);

		if (hasValue(currentConfig.typ_p)) apiOptions.typ_p = Number(currentConfig.typ_p);

		if (hasValue(currentConfig.repeat_last_n))
			apiOptions.repeat_last_n = Number(currentConfig.repeat_last_n);

		if (hasValue(currentConfig.repeat_penalty))
			apiOptions.repeat_penalty = Number(currentConfig.repeat_penalty);

		if (hasValue(currentConfig.presence_penalty))
			apiOptions.presence_penalty = Number(currentConfig.presence_penalty);

		if (hasValue(currentConfig.frequency_penalty))
			apiOptions.frequency_penalty = Number(currentConfig.frequency_penalty);

		if (hasValue(currentConfig.dry_multiplier))
			apiOptions.dry_multiplier = Number(currentConfig.dry_multiplier);

		if (hasValue(currentConfig.dry_base)) apiOptions.dry_base = Number(currentConfig.dry_base);

		if (hasValue(currentConfig.dry_allowed_length))
			apiOptions.dry_allowed_length = Number(currentConfig.dry_allowed_length);

		if (hasValue(currentConfig.dry_penalty_last_n))
			apiOptions.dry_penalty_last_n = Number(currentConfig.dry_penalty_last_n);

		if (currentConfig.samplers) apiOptions.samplers = currentConfig.samplers;

		if (hasValue(currentConfig.backend_sampling))
			apiOptions.backend_sampling = currentConfig.backend_sampling;

		if (currentConfig.customJson) apiOptions.custom = currentConfig.customJson;

		return apiOptions;
	}

	getChatStreaming(convId: string): { response: string; messageId: string } | undefined {
		return this.getChatStreamingState(convId);
	}

	async getDeletionInfo(messageId: string): Promise<{
		totalCount: number;
		userMessages: number;
		assistantMessages: number;
		messageTypes: string[];
	}> {
		return this.flows.getDeletionInfo(messageId);
	}

	getOrCreateAbortController(convId: string): AbortController {
		let c = this.abortControllers.get(convId);

		if (!c || c.signal.aborted) {
			c = new AbortController();
			this.abortControllers.set(convId, c);
		}

		return c;
	}

	getPendingMessageContent(convId: string): string | null {
		return this.pendingMessages.get(convId)?.content ?? null;
	}

	getPendingMessageExtras(convId: string): DatabaseMessageExtra[] | undefined {
		return this.pendingMessages.get(convId)?.extras;
	}

	getResumeModel(convId: string): string | null {
		return this.streams.getResumeModel(convId);
	}

	hasPendingDraft(): boolean {
		return Boolean(this.pendingDraftMessage) || this.pendingDraftFiles.length > 0;
	}

	hasPendingMessage(convId: string): boolean {
		return this.pendingMessages.has(convId);
	}

	injectPendingMessage(convId: string, content: string, extras?: DatabaseMessageExtra[]): void {
		this.pendingMessages.set(convId, { content, extras });
	}

	isChatLoading(convId: string): boolean {
		return this.activity.isLocal(convId);
	}

	isChatLoadingInternal(convId: string): boolean {
		return this.activity.isLocal(convId) || this.chatStreamingStates.has(convId);
	}

	isEditing(): boolean {
		return this.isEditModeActive;
	}

	/** True while the active conversation has a live streaming pipe. */
	isStreaming(): boolean {
		return this.chatStreamingStates.has(conversationsStore.activeConversation?.id ?? '');
	}

	/**
	 * Record a working-directory change into chat history as a synthetic
	 * user message, so the model sees it on its next turn (the client
	 * sends the cwd itself via the x-tool-cwd header on tool calls).
	 * A plain user message is used because some chat templates reject
	 * tool messages without a preceding tool call.
	 */
	async recordCwdChange(cwd: string | null): Promise<void> {
		const content = cwd
			? formatCwdMessage(cwd, await toolsStore.resolveServerHome())
			: CWD_CLEARED_TEXT;
		// Reuse the trailing cwd row when it is already the last message, so
		// repeated picks update it in place instead of stacking another row.
		const last = conversationsStore.activeMessages[conversationsStore.activeMessages.length - 1];

		if (last && last.role === MessageRole.USER && last.isSynthetic === true) {
			await DatabaseService.updateMessage(last.id, { content, isSynthetic: true });
			conversationsStore.updateMessageAtIndex(conversationsStore.activeMessages.length - 1, {
				content,
				isSynthetic: true
			});

			return;
		}

		await this.addMessage(MessageRole.USER, content, MessageType.TEXT, '-1', undefined, true);
	}

	async regenerateMessage(messageId: string): Promise<void> {
		return this.flows.regenerateMessage(messageId);
	}

	async regenerateMessageWithBranching(messageId: string, modelOverride?: string): Promise<void> {
		return this.flows.regenerateMessageWithBranching(messageId, modelOverride);
	}

	/**
	 * Opening the system message editor stores the message with the placeholder text; closing the
	 * editor other than with Cancel (switching chats, reloading) leaves it behind. Remove such a
	 * leftover when its conversation is opened again, unless it is the one being edited.
	 */
	async removeStalePlaceholder(): Promise<void> {
		const stale = conversationsStore.activeMessages.find(
			(m) =>
				m.role === MessageRole.SYSTEM &&
				m.content === SYSTEM_MESSAGE_PLACEHOLDER &&
				m.id !== this.pendingEditMessageId
		);

		if (stale) await this.removeSystemPromptPlaceholder(stale.id);
	}

	async removeSystemPromptPlaceholder(messageId: string): Promise<boolean> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) return false;

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const systemMessage = findMessageById(allMessages, messageId);

			if (!systemMessage || systemMessage.role !== MessageRole.SYSTEM) return false;

			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);

			if (!rootMessage) return false;

			if (allMessages.length === 2 && systemMessage.children.length === 0) {
				await conversationsStore.deleteConversation(activeConv.id);

				return true;
			}

			for (const childId of systemMessage.children) {
				await DatabaseService.updateMessage(childId, { parent: rootMessage.id });
				const childIndex = conversationsStore.findMessageIndex(childId);

				if (childIndex !== -1)
					conversationsStore.updateMessageAtIndex(childIndex, { parent: rootMessage.id });
			}
			await DatabaseService.updateMessage(rootMessage.id, {
				children: [
					...rootMessage.children.filter((id: string) => id !== messageId),
					...systemMessage.children
				]
			});
			await DatabaseService.deleteMessage(messageId);
			const systemIndex = conversationsStore.findMessageIndex(messageId);

			if (systemIndex !== -1) conversationsStore.activeMessages.splice(systemIndex, 1);

			conversationsStore.updateConversationTimestamp();

			return false;
		} catch (error) {
			console.error('Failed to remove system prompt placeholder:', error);

			return false;
		}
	}

	/**
	 * W1: output limits of a reply to `messages`: max_tokens = the configured limit capped at
	 * free - reserve (floor 1), enforced after the custom JSON, and a thinking budget of at most
	 * half of it. Empty when the reserve does not apply (off, or no context size).
	 */
	replyLimits(
		messages: readonly DatabaseMessage[],
		model?: string | null,
		convId?: string
	): { max_tokens?: number; maxTokensCeiling?: number; thinkingBudgetCap?: number } {
		const cfg = conversationsStore.getContextConfig(convId);
		const nCtx = this.contextSizeFor(model);

		if (!nCtx || cfg.reserve <= 0) return {};

		const max = Number(settingsStore.config.max_tokens);
		const cap = replyTokenCap({
			configured: max > 0 ? max : undefined,
			nCtx,
			reserve: cfg.reserve,
			used: measureContextUsage(messages, nCtx).used
		});

		if (cap === undefined) return {};

		return { max_tokens: cap, maxTokensCeiling: cap, thinkingBudgetCap: cappedThinkingBudget(cap) };
	}

	/** W1: does sending this message leave less than the reserve (+ a minimal reply) free? */
	reserveReached(convId: string, content: string, extras?: DatabaseMessageExtra[]): boolean {
		const cfg = conversationsStore.getContextConfig(convId);

		if (cfg.reserve <= 0) return false;

		if (conversationsStore.activeConversation?.id !== convId) return false;

		const messages = conversationsStore.activeMessages as DatabaseMessage[];
		const model = serverStore.isRouterMode
			? modelsStore.selectedModelName || getConversationModel(messages)
			: null;
		const nCtx = this.contextSizeFor(model);
		const usage = measureContextUsage(messages, nCtx);
		const pending = estimateMessageTokens({
			content,
			extra: extras,
			role: MessageRole.USER
		} as DatabaseMessage);

		return shouldCompact({ nCtx, pending, reserve: cfg.reserve, used: usage.used });
	}

	savePendingDraft(message: string, files: ChatUploadedFile[]): void {
		this.pendingDraftMessage = message;
		this.pendingDraftFiles = [...files];
	}

	/**
	 * W1: what a send of this message to the (open) conversation must do first: nothing, compact
	 * (the reserve would not stay free; auto-compaction on) or refuse (auto-compaction off - the
	 * reply could only eat into the reserve).
	 */
	sendGate(
		convId: string,
		content: string,
		extras?: DatabaseMessageExtra[]
	): 'send' | 'compact' | 'blocked' {
		if (!this.reserveReached(convId, content, extras)) return 'send';

		return conversationsStore.getContextConfig(convId).autoCompact ? 'compact' : 'blocked';
	}

	async sendMessage(content: string, extras?: DatabaseMessageExtra[]): Promise<void> {
		if (!content.trim() && (!extras || extras.length === 0)) return;

		const activeConv = conversationsStore.activeConversation;

		// If agentic loop is running, inject as a steering message instead of starting a new flow
		if (activeConv && agenticStore.isRunning(activeConv.id)) {
			agenticStore.injectSteeringMessage(activeConv.id, content, extras);

			return;
		}

		// If non-agentic streaming is active, queue as a pending message to send after completion
		if (activeConv && this.isChatLoadingInternal(activeConv.id)) {
			this.injectPendingMessage(activeConv.id, content, extras);

			return;
		}

		// Cancel any in-flight pre-encode request
		this.cancelPreEncode();

		// W1: a turn that would not leave the context reserve free compacts first and the message
		// goes to the continuation. It is bound to its conversation: when the compaction fails or is
		// cancelled, or the open conversation changed meanwhile, it is not sent anywhere else - it
		// stays the input draft of its own conversation. (The chat screen runs this gate before it
		// clears the input, so there a failed compaction just leaves the message in the box.)
		if (activeConv) {
			const gate = this.sendGate(activeConv.id, content, extras);

			if (gate === 'blocked') {
				toast.warning(
					'The context reserve is reached and auto-compaction is off: use "Compact now" first.'
				);
				this.keepAsDraft(activeConv.id, content, extras);

				return;
			}

			if (gate === 'compact') {
				const result = await this.compactConversation(activeConv.id, { auto: true });
				const target = 'convId' in result ? result.convId : activeConv.id;

				if (!('convId' in result) || conversationsStore.activeConversation?.id !== target) {
					this.keepAsDraft(target, content, extras);

					return;
				}
			}
		}

		// Consume MCP resource attachments - converts them to extras and clears the live store
		const resourceExtras = mcpStore.consumeResourceAttachmentsAsExtras();
		const allExtras = resourceExtras.length > 0 ? [...(extras || []), ...resourceExtras] : extras;

		let isNewConversation = false;

		if (!activeConv) {
			await conversationsStore.createConversation();
			isNewConversation = true;
		}

		const currentConv = conversationsStore.activeConversation;

		if (!currentConv) return;

		this.showErrorDialog(null);
		this.setChatLoading(currentConv.id, true);
		this.clearChatStreaming(currentConv.id);
		try {
			let parentIdForUserMessage: string | undefined;

			if (isNewConversation) {
				const rootId = await DatabaseService.createRootMessage(currentConv.id);
				const currentConfig = settingsStore.config;
				const systemPrompt = currentConfig.systemMessage?.toString().trim();

				let sysOrRootId = rootId;

				if (systemPrompt) {
					const systemMessage = await DatabaseService.createSystemMessage(
						currentConv.id,
						systemPrompt,
						rootId
					);

					conversationsStore.addMessageToActive(systemMessage);
					sysOrRootId = systemMessage.id;
				}

				// Reflect a working directory picked on the new-chat screen into
				// chat history before the first user message, so the model sees
				// it on its first turn. createConversation() has already threaded
				// the pending pick onto the conversation.
				if (currentConv.cwd) {
					const cwdMessage = await this.addMessage(
						MessageRole.USER,
						formatCwdMessage(currentConv.cwd, await toolsStore.resolveServerHome()),
						MessageType.TEXT,
						sysOrRootId,
						undefined,
						true
					);

					parentIdForUserMessage = cwdMessage.id;
				} else {
					parentIdForUserMessage = sysOrRootId;
				}
			}

			const userMessage = await this.addMessage(
				MessageRole.USER,
				content,
				MessageType.TEXT,
				parentIdForUserMessage ?? '-1',
				allExtras
			);

			if (isNewConversation && content)
				await conversationsStore.applyTitleFromContent(currentConv.id, content);

			const assistantMessage = await this.createAssistantMessage(userMessage.id);

			conversationsStore.addMessageToActive(assistantMessage);
			await this.streamChatCompletion(
				conversationsStore.activeMessages.slice(0, -1),
				assistantMessage,
				undefined,
				undefined,
				undefined,
				settingsStore.config.titleGenerationUseLLM && isNewConversation ? content : undefined
			);
		} catch (error) {
			if (isAbortError(error)) {
				this.setChatLoading(currentConv.id, false);

				return;
			}

			console.error('Failed to send message:', error);
			this.setChatLoading(currentConv.id, false);
			const dialogType =
				error instanceof Error && error.name === 'TimeoutError'
					? ErrorDialogType.TIMEOUT
					: ErrorDialogType.SERVER;
			const contextInfo = (
				error as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
			).contextInfo;

			this.showErrorDialog({
				contextInfo,
				message: error instanceof Error ? error.message : 'Unknown error',
				type: dialogType
			});
		}
	}

	setChatLoading(convId: string, loading: boolean): void {
		if (loading) {
			this.activity.markLocal(convId);
		} else {
			this.activity.localEnded(convId);
			this.setChatReasoning(convId, false);
		}
	}

	setChatReasoning(convId: string, reasoning: boolean): void {
		if (reasoning) this.chatReasoningStates.set(convId, true);
		else this.chatReasoningStates.delete(convId);
	}

	setChatStreaming(
		convId: string,
		response: string,
		messageId: string,
		model?: string | null
	): void {
		this.chatStreamingStates.set(convId, {
			messageId,
			model: model ?? this.chatStreamingStates.get(convId)?.model,
			response
		});

		if (convId === conversationsStore.activeConversation?.id) this.currentResponse = response;
	}
	setEditModeActive(handler: (files: File[]) => void): void {
		this.isEditModeActive = true;
		this.addFilesHandler = handler;
	}

	showErrorDialog(state: ErrorDialogState | null): void {
		this.errorDialogState = state;
	}

	async stopGeneration(): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) return;

		await this.stopGenerationForChat(activeConv.id);
	}

	async stopGenerationForChat(convId: string): Promise<void> {
		await this.savePartialResponseIfNeeded(convId);
		// tell the server to stop the generation, not just drop the HTTP socket. without this the
		// detached drain keeps producing tokens until eos or max_tokens. use the frozen identity
		// captured when the session started, not the live dropdown
		const streamStateForStop = this.chatStreamingStates.get(convId);
		const modelForStop = streamStateForStop?.model ?? ChatService.getStreamState(convId)?.model;

		void ChatService.cancelServerStream(convId, modelForStop);
		// an explicit stop leaves nothing to resume and kills a pending resume retry
		ChatService.clearStreamState(convId);
		this.streams.cancelResumeRetry(convId);
		this.abortRequest(convId);
		this.setChatLoading(convId, false);
		this.clearChatStreaming(convId);
		this.processing.setState(convId, null);
		this.clearPendingMessage(convId);
	}

	async streamChatCompletion(
		allMessages: DatabaseMessage[],
		assistantMessage: DatabaseMessage,
		onComplete?: (content: string) => Promise<void>,
		onError?: (error: Error) => void,
		modelOverride?: string | null,
		firstUserMessageContent?: string
	): Promise<void> {
		// the ::model suffix in the stream identity is only for router mode, where it routes to the
		// owning child. in single-model mode the identity stays the bare conv id so that attach, stop
		// and reattach all agree, regardless of fresh send vs regenerate passing a resolved model
		let effectiveModel: string | null | undefined = undefined;

		if (serverStore.isRouterMode) {
			const conversationModel = getConversationModel(allMessages);

			effectiveModel = modelOverride || modelsStore.selectedModelName || conversationModel;
		}

		if (serverStore.isRouterMode && effectiveModel) {
			if (!modelsStore.props.getModelProps(effectiveModel))
				await modelsStore.props.fetchModelProps(effectiveModel);
		}

		// Mutable state for the current message being streamed
		let currentMessageId = assistantMessage.id;
		let streamedContent = '';
		let streamedReasoningContent = '';
		let resolvedModel: string | null = null;
		let modelPersisted = false;

		const convId = assistantMessage.convId;

		// Tracks the last message created in this flow. Used as the parent for the next
		// turn's assistant message so createAssistantMessage does not have to read
		// conversationsStore.activeMessages, which may belong to a different conversation
		// after the user navigates while the loop is still running.
		let lastCreatedInFlow = currentMessageId;

		// freeze the POST identity from t0 so a stop cancels with the exact session key,
		// never a stale or empty model resolved later
		this.setChatStreaming(convId, streamedContent, currentMessageId, effectiveModel);

		const recordModel = (modelName: string | null | undefined, persistImmediately = true): void => {
			if (!modelName) return;

			const n = normalizeModelName(modelName);

			if (!n || n === resolvedModel) return;

			resolvedModel = n;
			const idx = conversationsStore.findMessageIndex(currentMessageId);

			conversationsStore.updateMessageAtIndex(idx, { model: n });

			if (persistImmediately && !modelPersisted) {
				modelPersisted = true;
				DatabaseService.updateMessage(currentMessageId, { model: n }).catch(() => {
					modelPersisted = false;
					resolvedModel = null;
				});
			}
		};

		let completionIdRecorded = false;

		const recordCompletionId = (id: string): void => {
			if (!id || completionIdRecorded) return;

			completionIdRecorded = true;
			const idx = conversationsStore.findMessageIndex(currentMessageId);

			conversationsStore.updateMessageAtIndex(idx, { completionId: id });
			DatabaseService.updateMessage(currentMessageId, { completionId: id }).catch(() => {
				completionIdRecorded = false;
			});
		};
		const updateStreamingUI = () => {
			this.setChatStreaming(convId, streamedContent, currentMessageId, effectiveModel);
			const idx = conversationsStore.findMessageIndex(currentMessageId);

			conversationsStore.updateMessageAtIndex(idx, { content: streamedContent });
		};
		const cleanupStreamingState = () => {
			this.setChatLoading(convId, false);
			this.clearChatStreaming(convId, currentMessageId);
			this.processing.setState(convId, null);
		};

		this.processing.setActiveConversation(convId);
		const abortController = this.getOrCreateAbortController(convId);
		const streamCallbacks: ChatStreamCallbacks = {
			createAssistantMessage: async () => {
				// Reset streaming state for new message
				streamedContent = '';
				streamedReasoningContent = '';

				const msg = await DatabaseService.createMessageBranch(
					{
						children: [],
						content: '',
						convId,
						model: resolvedModel,
						role: MessageRole.ASSISTANT,
						timestamp: Date.now(),
						toolCalls: '',
						type: MessageType.TEXT
					},
					lastCreatedInFlow
				);

				if (conversationsStore.activeConversation?.id === convId) {
					conversationsStore.addMessageToActive(msg);
				}

				currentMessageId = msg.id;
				lastCreatedInFlow = msg.id;

				return msg;
			},
			createToolResultMessage: async (
				toolCallId: string,
				content: string,
				extras?: DatabaseMessageExtra[],
				toolCwd?: string
			) => {
				const msg = await DatabaseService.createMessageBranch(
					{
						children: [],
						content,
						convId,
						extra: extras,
						// attachments stored once: the rebuild sends them as stored (no legacy repair)
						extraVersion: MESSAGE_EXTRA_STORAGE_VERSION,
						role: MessageRole.TOOL,
						timestamp: Date.now(),
						toolCallId,
						toolCalls: '',
						toolCwd,
						type: MessageType.TEXT
					},
					// chain parallel tool results (assistant -> tool 1 -> tool 2): parenting
					// each on the assistant made them siblings, so the path rebuilt on the
					// next turn kept only the last one
					lastCreatedInFlow
				);

				// mirror into the active store and move the node pointer only when this
				// conversation is displayed; otherwise persist the node move straight to
				// the db for the owning conv so a foreign conv's currNode stays untouched
				if (conversationsStore.activeConversation?.id === convId) {
					conversationsStore.addMessageToActive(msg);
					await conversationsStore.updateCurrentNode(msg.id);
				} else {
					await DatabaseService.updateCurrentNode(convId, msg.id);
				}

				lastCreatedInFlow = msg.id;

				return msg;
			},
			onAssistantTurnComplete: async (
				content: string,
				reasoningContent: string | undefined,
				timings: ChatMessageTimings | undefined,
				toolCalls: import('$lib/types/api').ApiChatCompletionToolCall[] | undefined
			) => {
				const updateData: Record<string, unknown> = {
					content,
					reasoningContent: reasoningContent || undefined,
					timings,
					toolCalls: toolCalls ? JSON.stringify(toolCalls) : ''
				};

				if (resolvedModel && !modelPersisted) updateData.model = resolvedModel;

				await DatabaseService.updateMessage(currentMessageId, updateData);
				const idx = conversationsStore.findMessageIndex(currentMessageId);
				const uiUpdate: Partial<DatabaseMessage> = {
					content,
					reasoningContent: reasoningContent || undefined,
					toolCalls: toolCalls ? JSON.stringify(toolCalls) : ''
				};

				if (timings) uiUpdate.timings = timings;

				if (resolvedModel) uiUpdate.model = resolvedModel;

				// touch the active ui array and node pointer only when this conversation
				// is displayed; otherwise persist the node move straight to the db so a
				// foreign conv's currNode stays untouched
				if (conversationsStore.activeConversation?.id === convId) {
					conversationsStore.updateMessageAtIndex(idx, uiUpdate);
					await conversationsStore.updateCurrentNode(currentMessageId);
				} else {
					await DatabaseService.updateCurrentNode(convId, currentMessageId);
				}
			},
			onChunk: (chunk: string) => {
				streamedContent += chunk;
				updateStreamingUI();
				this.setChatReasoning(convId, false);
			},
			onCompletionId: (id: string) => recordCompletionId(id),
			onError: async (error: Error) => {
				if (isAbortError(error)) {
					cleanupStreamingState();
					// If aborted with a pending message (e.g. "Send immediately"), re-send it
					const pending = this.consumePendingMessage(convId);

					if (pending) {
						this.sendMessage(pending.content, pending.extras);
					}

					return;
				}

				console.error('Streaming error:', error);
				// keep whatever was streamed so far, the message stays in memory and in DB
				await this.savePartialResponseIfNeeded(convId);
				cleanupStreamingState();
				this.clearPendingMessage(convId);

				const contextInfo = (
					error as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
				).contextInfo;

				this.showErrorDialog({
					contextInfo,
					message: error.message,
					type: error.name === 'TimeoutError' ? ErrorDialogType.TIMEOUT : ErrorDialogType.SERVER
				});

				if (onError) onError(error);
			},
			onFlowComplete: (finalTimings?: ChatMessageTimings) => {
				if (finalTimings) {
					const idx = conversationsStore.findMessageIndex(assistantMessage.id);

					conversationsStore.updateMessageAtIndex(idx, { timings: finalTimings });
					DatabaseService.updateMessage(assistantMessage.id, {
						timings: finalTimings
					}).catch(console.error);
				}

				cleanupStreamingState();

				if (onComplete) onComplete(streamedContent);

				if (serverStore.isRouterMode) modelsStore.fetchRouterModels().catch(console.error);

				// Pre-encode conversation in KV cache for faster next turn
				if (settingsStore.config.preEncodeConversation) {
					this.triggerPreEncode(convId, effectiveModel);
				}
			},
			onModel: (modelName: string) => recordModel(modelName),
			onReasoningChunk: (chunk: string) => {
				streamedReasoningContent += chunk;
				// mark streaming state so a stop mid-thinking can persist the partial reasoning
				this.setChatStreaming(convId, streamedContent, currentMessageId, effectiveModel);
				const idx = conversationsStore.findMessageIndex(currentMessageId);

				conversationsStore.updateMessageAtIndex(idx, {
					reasoningContent: streamedReasoningContent
				});
				this.setChatReasoning(convId, true);
			},
			onTimings: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => {
				this.processing.applyStreamTimings(timings, promptProgress, convId);
			},
			onToolCallsStreaming: (toolCalls) => {
				const idx = conversationsStore.findMessageIndex(currentMessageId);

				conversationsStore.updateMessageAtIndex(idx, {
					toolCalls: JSON.stringify(toolCalls)
				});
			},
			onTurnComplete: (intermediateTimings: ChatMessageTimings) => {
				// Update the first assistant message with cumulative agentic timings
				const idx = conversationsStore.findMessageIndex(assistantMessage.id);

				conversationsStore.updateMessageAtIndex(idx, { timings: intermediateTimings });
			},
			updateToolResultMessage: async (
				messageId: string,
				content: string,
				extras?: DatabaseMessageExtra[]
			) => {
				// Persist latest content + merged extras; mirror into the active
				// store so the chat view sees live updates for streaming tools
				// (e.g. exec_shell_command). The existing tool message node
				// pointer stays put - the renderer is already scoped to it.
				const updates: Partial<DatabaseMessage> = { content };

				if (extras) {
					const idx = conversationsStore.findMessageIndex(messageId);
					const existing = idx >= 0 ? (conversationsStore.activeMessages[idx]?.extra ?? []) : [];
					const merged = [...existing, ...extras];

					updates.extra = merged;
				}

				if (conversationsStore.activeConversation?.id === convId) {
					const idx = conversationsStore.findMessageIndex(messageId);

					if (idx >= 0) conversationsStore.updateMessageAtIndex(idx, updates);
				}

				await DatabaseService.updateMessage(messageId, updates);
			}
		};
		const toolPolicy = {
			disabledToolCategories: conversationsStore.preferences.getDisabledToolCategories(),
			disabledTools: conversationsStore.preferences.getDisabledTools()
		};
		// W1: the reply never uses the context reserve (the agentic loop re-caps each later turn)
		const replyCap = this.replyLimits(allMessages, effectiveModel, convId);

		{
			const agenticResult = await agenticStore.runAgenticFlow({
				callbacks: streamCallbacks,
				conversationId: convId,
				flowRootMessageId: assistantMessage.id,
				messages: allMessages,
				options: {
					...this.getApiOptions(),
					...(effectiveModel ? { model: effectiveModel } : {}),
					...replyCap
				},
				signal: abortController.signal,
				toolPolicy
			});

			if (agenticResult.handled) {
				// Generate LLM based title for new conversations after agentic flow completes
				if (firstUserMessageContent) {
					await this.generateTitleWithLLM(firstUserMessageContent, streamedContent, convId);
				}

				// Check if there's a pending steering message to re-send
				const pending = agenticStore.consumePendingSteeringMessage(convId);

				if (pending) {
					await this.sendMessage(pending.content, pending.extras);
				}

				return;
			}
		}

		await ChatService.sendMessage(
			allMessages,
			{
				...this.getApiOptions(),
				...(effectiveModel ? { model: effectiveModel } : {}),
				...replyCap,
				onChunk: streamCallbacks.onChunk,
				onComplete: async (
					finalContent?: string,
					reasoningContent?: string,
					timings?: ChatMessageTimings,
					toolCalls?: string
				) => {
					const content = streamedContent || finalContent || '';
					const reasoning = streamedReasoningContent || reasoningContent;
					const updateData: Record<string, unknown> = {
						content,
						reasoningContent: reasoning || undefined,
						timings,
						toolCalls: toolCalls || ''
					};

					if (resolvedModel && !modelPersisted) updateData.model = resolvedModel;

					await DatabaseService.updateMessage(currentMessageId, updateData);
					const idx = conversationsStore.findMessageIndex(currentMessageId);
					const uiUpdate: Partial<DatabaseMessage> = {
						content,
						reasoningContent: reasoning || undefined,
						toolCalls: toolCalls || ''
					};

					if (timings) uiUpdate.timings = timings;

					if (resolvedModel) uiUpdate.model = resolvedModel;

					conversationsStore.updateMessageAtIndex(idx, uiUpdate);
					await conversationsStore.updateCurrentNode(currentMessageId);
					cleanupStreamingState();

					if (onComplete) await onComplete(content);

					if (serverStore.isRouterMode) modelsStore.fetchRouterModels().catch(console.error);

					// Generate LLM based title for new conversations (avoids stale reference
					// issue when user switches conversations while streaming)
					if (firstUserMessageContent) {
						await this.generateTitleWithLLM(firstUserMessageContent, streamedContent, convId);
					}

					// Check if there's a pending message queued during streaming
					const pending = this.consumePendingMessage(convId);

					if (pending) {
						await this.sendMessage(pending.content, pending.extras);
					}
				},
				onCompletionId: streamCallbacks.onCompletionId,
				onConnectionState: (state: StreamConnectionState) => {
					if (convId === conversationsStore.activeConversation?.id) {
						this.streamConnectionState = state;
					}
				},
				onError: streamCallbacks.onError,
				onModel: streamCallbacks.onModel,
				onReasoningChunk: streamCallbacks.onReasoningChunk,
				onTimings: streamCallbacks.onTimings,
				stream: true
			},
			convId,
			abortController.signal
		);
	}

	syncLoadingStateForChat(convId: string): void {
		const s = this.chatStreamingStates.get(convId);

		this.currentResponse = s?.response || '';
		this.processing.setActiveConversation(convId);

		// Sync streaming content to activeMessages so UI displays current content
		if (s?.response && s?.messageId) {
			const idx = conversationsStore.findMessageIndex(s.messageId);

			if (idx !== -1) {
				conversationsStore.updateMessageAtIndex(idx, { content: s.response });
			}
		}
	}

	async syncRemoteRunningStreams(): Promise<void> {
		return this.streams.syncRemoteRunningStreams();
	}

	/**
	 * Message flows (edit / regenerate / continue / delete) live in
	 * ChatMessageFlows; these delegate so consumers keep a single entry point.
	 */
	async updateMessage(messageId: string, newContent: string): Promise<void> {
		return this.flows.updateMessage(messageId, newContent);
	}

	private abortRequest(convId?: string): void {
		if (convId) {
			const c = this.abortControllers.get(convId);

			if (c) {
				c.abort();
				this.abortControllers.delete(convId);
			}
		} else {
			for (const c of this.abortControllers.values()) c.abort();
			this.abortControllers.clear();
		}
	}

	private async generateTitleWithLLM(
		userContent: string,
		assistantContent: string,
		convId: string
	): Promise<void> {
		const effectiveModel =
			serverStore.isRouterMode && modelsStore.selectedModelName
				? modelsStore.selectedModelName
				: undefined;
		const configValue = settingsStore.config;
		const titlePromptTemplate =
			typeof configValue.titleGenerationPrompt === 'string' &&
			configValue.titleGenerationPrompt.trim()
				? configValue.titleGenerationPrompt
				: TITLE_GENERATION.DEFAULT_PROMPT;
		const titlePrompt = titlePromptTemplate
			.replace('{{USER}}', String(userContent || ''))
			.replace('{{ASSISTANT}}', String(assistantContent || ''));
		const titleMessage: ApiChatMessageData = {
			content: titlePrompt,
			role: MessageRole.USER
		};
		const titleResponse = await ChatService.generateTitle(titleMessage, effectiveModel);

		if (!titleResponse) {
			return;
		}

		let cleanTitle = titleResponse.trim();

		cleanTitle = cleanTitle
			.replace(TITLE_GENERATION.PREFIX_PATTERN, '')
			.replace(TITLE_GENERATION.QUOTE_PATTERN, '')
			.trim();

		if (!cleanTitle || cleanTitle.length < TITLE_GENERATION.MIN_LENGTH) {
			const firstLine = userContent.split('\n').find((l) => l.trim().length > 0);

			cleanTitle = firstLine ? firstLine.trim() : TITLE_GENERATION.FALLBACK;
		}

		if (cleanTitle && cleanTitle.length >= TITLE_GENERATION.MIN_LENGTH) {
			await conversationsStore.updateConversationName(convId, cleanTitle);
		}
	}

	private getChatStreamingState(
		convId: string
	): { response: string; messageId: string } | undefined {
		return this.chatStreamingStates.get(convId);
	}

	/** W1: keep an unsent message as the input draft of its own conversation. */
	private keepAsDraft(convId: string, content: string, extras?: DatabaseMessageExtra[]): void {
		draftMessagesStore.saveDraftMessage(convId, content, []);
		toast.info(
			extras && extras.length > 0
				? 'The message was not sent; its text is kept as the draft of its conversation (attachments must be added again).'
				: 'The message was not sent; it is kept as the draft of its conversation.'
		);
	}
	private async runCompaction(
		convId: string,
		owner: CompactionState,
		abortController: AbortController,
		fail: (error: string) => CompactionResult,
		opts: { auto?: boolean }
	): Promise<CompactionResult> {
		const conversation = await DatabaseService.getConversation(convId);

		if (!conversation?.currNode) return fail('the conversation is empty');

		const path = dropTrailingEmptyAssistants(
			filterByLeafNodeId(
				await DatabaseService.getConversationMessages(convId),
				conversation.currNode,
				false
			) as DatabaseMessage[]
		).filter((m) => m.type !== MessageType.ROOT);
		const model = serverStore.isRouterMode
			? modelsStore.selectedModelName || getConversationModel(path)
			: null;
		const nCtx = this.contextSizeFor(model);

		if (!nCtx) return fail('the context size is unknown (no /props yet)');

		const cfg = resolveContextReserveConfig(settingsStore.config, conversation.contextSettings);
		const reserve = cfg.reserve > 0 ? cfg.reserve : CONTEXT_RESERVE.DEFAULT_RESERVE_TOKENS;
		const instructionTokens =
			estimateTextTokens(buildSummaryInstruction(cfg.summaryPrompt, reserve)) +
			CONTEXT_RESERVE.MESSAGE_OVERHEAD_TOKENS * 2;

		if (!reserveHoldsSummary(reserve, instructionTokens)) {
			return fail(
				`the context reserve (${reserve}) minus the instruction (~${instructionTokens}) is below ` +
					`the ${CONTEXT_RESERVE.MIN_SUMMARY_TOKENS}-token minimum summary; raise the reserve`
			);
		}

		const plan = planSummaryRequest({ instructionTokens, messages: path, nCtx, reserve });

		if (!plan) return fail('no point of the conversation leaves room for a summary');

		const instruction = buildSummaryInstruction(cfg.summaryPrompt, plan.maxTokens);
		const instructionMessage = {
			children: [],
			content: instruction,
			convId,
			id: `compact-instruction-${convId}`,
			parent: null,
			role: MessageRole.USER,
			timestamp: Date.now(),
			toolCalls: '',
			type: MessageType.TEXT
		} as DatabaseMessage;
		// the request of a real turn (options of this conversation, its frozen tool list), so the
		// prompt is a pure extension of the cached one
		const frozenTools = parseToolsSnapshot(conversation.toolsSnapshot);
		const tools =
			agenticStore.getConfig(settingsStore.config).enabled && frozenTools && frozenTools.length > 0
				? frozenTools
				: undefined;

		let summary = '';
		let summaryModel: string | null = null;
		let completion: { finishReason?: string; toolCalls?: string } | null = null;

		await ChatService.sendMessage(
			[...path.slice(0, plan.keep), instructionMessage],
			{
				...this.getApiOptions(conversation),
				...(model ? { model } : {}),
				max_tokens: plan.maxTokens,
				// enforced after the custom JSON
				maxTokensCeiling: plan.maxTokens,
				onChunk: (chunk: string) => {
					summary += chunk;
				},
				onComplete: (
					_c: string,
					_r?: string,
					_t?: ChatMessageTimings,
					toolCalls?: string,
					finishReason?: string
				) => {
					completion = { finishReason, toolCalls };
				},
				onError: () => {},
				onModel: (name: string) => {
					summaryModel = normalizeModelName(name) || summaryModel;
				},
				onTimings: (timings?: ChatMessageTimings) => {
					if (this.compaction?.id === owner.id && timings?.predicted_n !== undefined) {
						this.compaction = { ...owner, tokens: timings.predicted_n };
					}
				},
				stream: true,
				// thinking may take at most a quarter of the summary budget
				thinkingBudgetCap: Math.floor(plan.maxTokens / 4),
				tools
			},
			undefined,
			abortController.signal
		);

		if (abortController.signal.aborted) return { cancelled: true };

		const done = completion as { finishReason?: string; toolCalls?: string } | null;
		const rejection = summaryRejection({
			completed: done !== null,
			finishReason: done?.finishReason,
			text: summary,
			toolCalls: done?.toolCalls
		});

		if (rejection) return fail(rejection);

		const newConvId = await this.saveCompactedConversation(conversation, path, plan.cut, {
			keepMessages: cfg.keepMessages,
			model: summaryModel,
			nCtx,
			reserve,
			summary: summary.trim()
		});

		await conversationsStore.loadConversations();
		await conversationsStore.loadConversation(newConvId);
		await goto(RouterService.chat(newConvId));

		if (!opts.auto) toast.success('Conversation compacted; the previous one is kept');

		return { convId: newConvId };
	}

	/** W1: write the compacted conversation; returns its id. */
	private async saveCompactedConversation(
		source: DatabaseConversation,
		path: readonly DatabaseMessage[],
		cut: boolean,
		p: {
			keepMessages: number;
			model: string | null;
			nCtx: number;
			reserve: number;
			summary: string;
		}
	): Promise<string> {
		const system = path.find(
			(m) =>
				m.role === MessageRole.SYSTEM &&
				m.content.trim() &&
				m.content.trim() !== SYSTEM_MESSAGE_PLACEHOLDER
		);
		const body = path.filter((m) => m.role !== MessageRole.SYSTEM && !m.isSynthetic);
		// the working directory row the model last saw (the tail carries no synthetic rows)
		const cwdRow = [...path].reverse().find((m) => m.isSynthetic && m.role === MessageRole.USER);
		// at most a quarter of the context, and what the new chat has left after the reserve, the
		// summary note, the head rows and a minimal reply
		const tail = selectTailMessages(
			body,
			p.keepMessages,
			tailTokenBudget({
				headTokens:
					(system ? estimateMessageTokens(system) : 0) +
					(cwdRow ? estimateMessageTokens(cwdRow) : 0),
				nCtx: p.nCtx,
				reserve: p.reserve,
				summaryTokens: estimateTextTokens(p.summary) + 64
			})
		);
		const newConv = await DatabaseService.createConversation(
			compactedConversationName(source.name),
			{
				compactedFromConversationId: source.id,
				// per-chat Context settings (W1.1) carry over to the continuation
				contextSettings: source.contextSettings,
				cwd: source.cwd,
				disabledToolCategories: source.disabledToolCategories,
				disabledTools: source.disabledTools,
				reasoningEffort: source.reasoningEffort,
				thinkingEnabled: source.thinkingEnabled,
				toolsDeclinedKey: source.toolsDeclinedKey,
				toolsSnapshot: source.toolsSnapshot
			}
		);
		const rootId = await DatabaseService.createRootMessage(newConv.id);

		let parent = rootId;

		if (system) {
			parent = (await DatabaseService.createSystemMessage(newConv.id, system.content, rootId)).id;
		}

		const copy = async (m: Omit<DatabaseMessage, 'id' | 'convId' | 'parent' | 'children'>) => {
			const created = await DatabaseService.createMessageBranch(
				{ ...m, children: [], convId: newConv.id, parent } as Omit<DatabaseMessage, 'id'>,
				parent
			);

			parent = created.id;
		};

		if (cwdRow) {
			await copy({
				content: cwdRow.content,
				isSynthetic: true,
				role: MessageRole.USER,
				timestamp: Date.now(),
				toolCalls: '',
				type: MessageType.TEXT
			});
		}

		const header = CONTEXT_RESERVE.SUMMARY_HEADER.replace('{{NAME}}', source.name);

		await copy({
			content: `${header}${cut ? ' (the end of the last turn did not fit in the summary request)' : ''}\n\n${p.summary}`,
			model: p.model ?? undefined,
			role: MessageRole.ASSISTANT,
			timestamp: Date.now(),
			toolCalls: '',
			type: MessageType.TEXT
		});

		for (const m of tail) {
			// measurements belong to the old context: the new one measures itself on its first reply
			// eslint-disable-next-line @typescript-eslint/no-unused-vars
			const { children, completionId, convId, id, parent: _p, timings, ...rest } = m;

			await copy({ ...rest, timestamp: Date.now() });
		}

		await DatabaseService.updateConversation(source.id, {
			compactedIntoConversationId: newConv.id
		});

		return newConv.id;
	}

	private async savePartialResponseIfNeeded(convId?: string): Promise<void> {
		const conversationId = convId || conversationsStore.activeConversation?.id;

		if (!conversationId) return;

		const streamingState = this.getChatStreamingState(conversationId);

		if (!streamingState) return;

		const messages =
			conversationId === conversationsStore.activeConversation?.id
				? conversationsStore.activeMessages
				: await conversationsStore.getConversationMessages(conversationId);

		if (!messages.length) return;

		const lastMessage = messages[messages.length - 1];

		if (lastMessage?.role !== MessageRole.ASSISTANT) return;

		const partialContent = streamingState.response;
		const partialReasoning = lastMessage.reasoningContent || '';
		// snapshot the streamed tool calls before clearing so we still know whether
		// anything was captured when deciding to skip the DB write below
		const hadPartialToolCalls = !!lastMessage.toolCalls?.trim();

		// nothing to persist when content, reasoning, and streamed tool calls are all empty
		// (e.g. stop before any token). otherwise drop the partial tool call and write whatever
		// was streamed: incomplete arguments (truncated JSON, missing closing quote) would
		// otherwise be re-sent to the server on the next turn and rejected.
		if (!partialContent.trim() && !partialReasoning.trim() && !hadPartialToolCalls) return;

		try {
			const updateData: {
				content?: string;
				reasoningContent?: string;
				toolCalls?: string;
				timings?: ChatMessageTimings;
			} = {
				toolCalls: ''
			};

			if (partialContent.trim()) updateData.content = partialContent;

			if (partialReasoning.trim()) updateData.reasoningContent = partialReasoning;

			const lastKnownState = this.processing.getState(conversationId);

			if (lastKnownState) {
				updateData.timings = {
					cache_n: lastKnownState.cacheTokens || 0,
					predicted_ms:
						lastKnownState.tokensPerSecond && lastKnownState.tokensDecoded
							? (lastKnownState.tokensDecoded / lastKnownState.tokensPerSecond) * 1000
							: undefined,
					predicted_n: lastKnownState.tokensDecoded || 0,
					prompt_ms: lastKnownState.promptMs,
					prompt_n: lastKnownState.promptTokens || 0
				};
			}

			await DatabaseService.updateMessage(lastMessage.id, updateData);
			lastMessage.content = partialContent;
			// mirror the drop into the in-memory message so the next request sent via
			// sendMessage (queued pending, Send immediately, or manual follow-up) reads
			// the cleared value, not whatever the streaming widget had been showing
			lastMessage.toolCalls = '';

			if (updateData.timings) lastMessage.timings = updateData.timings;
		} catch (error) {
			lastMessage.content = partialContent;
			lastMessage.toolCalls = '';
			console.error('Failed to save partial response:', error);
		}
	}

	/**
	 * W1: a message queued (typed while the summary ran) belongs to the conversation it was typed
	 * in: after a successful compaction it goes to the continuation (sent now after a manual
	 * compaction when that conversation is open; after the triggering reply after an automatic
	 * one); otherwise it is kept as the input draft of its conversation - never sent into
	 * whatever conversation happens to be open.
	 */
	private settleQueuedAfterCompaction(
		convId: string,
		result: CompactionResult,
		auto: boolean
	): void {
		const pending = this.consumePendingMessage(convId);

		if (!pending) return;

		const target = 'convId' in result ? result.convId : convId;
		const active = conversationsStore.activeConversation?.id;

		if ('convId' in result && active === target) {
			if (auto) this.pendingMessages.set(target, pending);
			else void this.sendMessage(pending.content, pending.extras);

			return;
		}

		this.keepAsDraft(target, pending.content, pending.extras);
	}

	private async triggerPreEncode(convId: string, model?: string | null): Promise<void> {
		this.cancelPreEncode();
		this.preEncodeAbortController = new AbortController();

		const signal = this.preEncodeAbortController.signal;

		try {
			const allIdle = await ChatService.areAllSlotsIdle(model, signal);

			if (!allIdle || signal.aborted) return;

			// the path the next request rebuilds from the database (tool calls, tool results,
			// reasoning), with the request options of a real turn and the frozen tool list;
			// conversation-specific options come from this conversation's own record, never
			// from the active one (the user may have switched conversations meanwhile)
			const conversation = await DatabaseService.getConversation(convId);

			if (!conversation) return;

			const path = filterByLeafNodeId(
				await DatabaseService.getConversationMessages(convId),
				conversation.currNode,
				false
			) as DatabaseMessage[];
			const frozenTools = parseToolsSnapshot(conversation.toolsSnapshot);

			await ChatService.preEncode(
				path,
				{
					...this.getApiOptions(conversation),
					...(model ? { model } : {}),
					tools: frozenTools ?? undefined
				},
				signal
			);
		} catch (err) {
			if (!isAbortError(err)) {
				console.warn('[ChatStore] Pre-encode failed:', err);
			}
		}
	}
}

export const chatStore = new ChatStore();
