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

	let useGlobalReserve = $state(true);
	let useGlobalAuto = $state(true);
	let useGlobalKeep = $state(true);
	let useGlobalPrompt = $state(true);
	let reserve = $state('');
	let autoCompact = $state(true);
	let keep = $state('');
	let prompt = $state('');

	// load the stored overrides each time the dialog opens
	$effect(() => {
		if (!open) return;

		const conv =
			conversationsStore.activeConversation?.id === convId
				? conversationsStore.activeConversation
				: conversationsStore.conversations.find((c) => c.id === convId);
		const o: ConversationContextSettings = conv?.contextSettings ?? {};

		useGlobalReserve = o.reserve === undefined;
		useGlobalAuto = o.autoCompact === undefined;
		useGlobalKeep = o.keepMessages === undefined;
		useGlobalPrompt = o.summaryPrompt === undefined;
		reserve = String(o.reserve ?? global.reserve);
		autoCompact = o.autoCompact ?? global.autoCompact;
		keep = String(o.keepMessages ?? global.keepMessages);
		prompt = o.summaryPrompt ?? global.summaryPrompt;
	});

	function toInt(value: string): number | undefined {
		const n = Number(value);

		return value.trim() !== '' && Number.isFinite(n) && n >= 0 ? Math.floor(n) : undefined;
	}

	async function save() {
		await conversationsStore.setContextSettings(convId, {
			autoCompact: useGlobalAuto ? undefined : autoCompact,
			keepMessages: useGlobalKeep ? undefined : toInt(keep),
			reserve: useGlobalReserve ? undefined : toInt(reserve),
			summaryPrompt: useGlobalPrompt ? undefined : prompt
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
				continuations.
			</Dialog.Description>
		</Dialog.Header>

		<div class="space-y-4 text-sm">
			<div class="space-y-1">
				<div class="flex items-center justify-between gap-2">
					<label class="font-medium" for="ctx-reserve">Context reserve (tokens)</label>

					<label class="flex items-center gap-1.5 text-xs text-muted-foreground">
						<Checkbox
							checked={useGlobalReserve}
							onCheckedChange={(v) => (useGlobalReserve = v === true)}
						/>
						use global default ({global.reserve})
					</label>
				</div>

				<Input
					bind:value={reserve}
					disabled={useGlobalReserve}
					id="ctx-reserve"
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
					<label class="font-medium" for="ctx-keep">Messages kept after compaction</label>

					<label class="flex items-center gap-1.5 text-xs text-muted-foreground">
						<Checkbox
							checked={useGlobalKeep}
							onCheckedChange={(v) => (useGlobalKeep = v === true)}
						/>
						use global default ({global.keepMessages})
					</label>
				</div>

				<Input bind:value={keep} disabled={useGlobalKeep} id="ctx-keep" min="0" type="number" />
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
					{'{{MAX_TOKENS}}'} is replaced by the summary length limit.
				</p>
			</div>
		</div>

		<Dialog.Footer>
			<Button onclick={() => (open = false)} type="button" variant="outline">Cancel</Button>

			<Button onclick={save} type="button">Save</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
