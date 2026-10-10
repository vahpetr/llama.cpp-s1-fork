<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import { Textarea } from '$lib/components/ui/textarea';
	import { conversationsStore, settingsStore } from '$lib/stores';
	import type { ConversationContextSettings } from '$lib/types';
	import { readContextReserveConfig } from '$lib/utils';

	// W1.1: the Context settings of one conversation; each field overrides the global setting or
	// follows it ("use global default")

	interface Props {
		open: boolean;
		convId: string;
	}

	let { convId, open = $bindable() }: Props = $props();

	let global = $derived(readContextReserveConfig(settingsStore.config));

	let useGlobalSummary = $state(true);
	let useGlobalAuto = $state(true);
	let useGlobalTail = $state(true);
	let useGlobalPrompt = $state(true);
	let summaryTokens = $state('');
	let autoCompact = $state(true);
	let tailTokens = $state('');
	let prompt = $state('');

	// load the stored overrides each time the dialog opens
	$effect(() => {
		if (!open) return;

		const conv =
			conversationsStore.activeConversation?.id === convId
				? conversationsStore.activeConversation
				: conversationsStore.conversations.find((c) => c.id === convId);
		const o: ConversationContextSettings = conv?.contextSettings ?? {};

		// W1.1's reserve / keepMessages overrides are not shown; saving drops them
		useGlobalSummary = o.summaryTokens === undefined;
		useGlobalAuto = o.autoCompact === undefined;
		useGlobalTail = o.tailTokens === undefined;
		useGlobalPrompt = o.summaryPrompt === undefined;
		summaryTokens = String(o.summaryTokens ?? global.summaryTokens);
		autoCompact = o.autoCompact ?? global.autoCompact;
		tailTokens = String(o.tailTokens ?? global.tailTokens);
		prompt = o.summaryPrompt ?? global.summaryPrompt;
	});

	function toInt(value: string): number | undefined {
		const n = Number(value);

		return value.trim() !== '' && Number.isFinite(n) && n >= 0 ? Math.floor(n) : undefined;
	}

	async function save() {
		await conversationsStore.setContextSettings(convId, {
			autoCompact: useGlobalAuto ? undefined : autoCompact,
			summaryPrompt: useGlobalPrompt ? undefined : prompt,
			summaryTokens: useGlobalSummary ? undefined : toInt(summaryTokens),
			tailTokens: useGlobalTail ? undefined : toInt(tailTokens)
		});
		open = false;
	}
</script>

<Dialog.Root bind:open>
	<Dialog.Content class="max-w-xl">
		<Dialog.Header>
			<Dialog.Title>Context settings for this chat</Dialog.Title>

			<Dialog.Description>
				Overrides of the global Context settings, for this conversation and its compacted
				continuations. The reserve is derived: summary length + instruction + 10 % of the context in
				use (at least 2048); the conversation compacts before a message when less than the reserve +
				32768 tokens are free.
			</Dialog.Description>
		</Dialog.Header>

		<div class="space-y-4 text-sm">
			<div class="space-y-1">
				<div class="flex items-center justify-between gap-2">
					<label class="font-medium" for="ctx-summary">Summary length (tokens)</label>

					<label class="flex items-center gap-1.5 text-xs text-muted-foreground">
						<Checkbox
							checked={useGlobalSummary}
							onCheckedChange={(v) => (useGlobalSummary = v === true)}
						/>
						use global default ({global.summaryTokens})
					</label>
				</div>

				<Input
					bind:value={summaryTokens}
					disabled={useGlobalSummary}
					id="ctx-summary"
					min="0"
					type="number"
				/>
			</div>

			<div class="flex items-center justify-between gap-2">
				<label class="flex items-center gap-1.5 font-medium">
					<Checkbox
						checked={autoCompact}
						disabled={useGlobalAuto}
						onCheckedChange={(v) => (autoCompact = v === true)}
					/>
					Auto-compact
				</label>

				<label class="flex items-center gap-1.5 text-xs text-muted-foreground">
					<Checkbox checked={useGlobalAuto} onCheckedChange={(v) => (useGlobalAuto = v === true)} />
					use global default ({global.autoCompact ? 'on' : 'off'})
				</label>
			</div>

			<div class="space-y-1">
				<div class="flex items-center justify-between gap-2">
					<label class="font-medium" for="ctx-tail">Kept tail (tokens)</label>

					<label class="flex items-center gap-1.5 text-xs text-muted-foreground">
						<Checkbox
							checked={useGlobalTail}
							onCheckedChange={(v) => (useGlobalTail = v === true)}
						/>
						use global default ({global.tailTokens})
					</label>
				</div>

				<Input
					bind:value={tailTokens}
					disabled={useGlobalTail}
					id="ctx-tail"
					min="0"
					type="number"
				/>
			</div>

			<div class="space-y-1">
				<div class="flex items-center justify-between gap-2">
					<label class="font-medium" for="ctx-prompt">Summary instruction</label>

					<label class="flex items-center gap-1.5 text-xs text-muted-foreground">
						<Checkbox
							checked={useGlobalPrompt}
							onCheckedChange={(v) => (useGlobalPrompt = v === true)}
						/>
						use global default
					</label>
				</div>

				<Textarea
					bind:value={prompt}
					class="min-h-[8rem]"
					disabled={useGlobalPrompt}
					id="ctx-prompt"
				/>

				<p class="text-xs text-muted-foreground">
					{'{{MAX_TOKENS}}'} is replaced by the summary length.
				</p>
			</div>
		</div>

		<Dialog.Footer>
			<Button onclick={() => (open = false)} type="button" variant="outline">Cancel</Button>

			<Button onclick={save} type="button">Save</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
