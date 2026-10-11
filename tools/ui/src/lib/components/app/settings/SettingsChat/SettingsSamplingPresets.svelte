<script lang="ts">
	// Settings -> Sampling & Penalties: the sampling presets by reasoning state (Thinking for an
	// explicit level, Instruct for Off) and optional per-level overrides of Thinking. Edits go to the
	// panel's draft (`presets`), persisted with Save and dropped by Cancel / close; the user's own
	// Sampling fields above always win; Default sends none.
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import {
		activePresetLine,
		PRESET_FIELDS,
		type PresetField,
		presetValues,
		SAMPLING_PRESETS,
		type SamplingPresetSettings,
		sanitizePresetValue,
		withoutPreset,
		withPresetField
	} from '$lib/constants';
	import { useReasoningMenu } from '$lib/hooks/use-reasoning-menu.svelte';

	interface Props {
		presets: SamplingPresetSettings;
	}

	let { presets = $bindable() }: Props = $props();

	// the same capability / level derivation as the picker (tracks the props cache version)
	const reasoning = useReasoningMenu();

	const FIELD_LABELS: Record<PresetField, string> = {
		min_p: 'Min P',
		presence_penalty: 'Presence penalty',
		repeat_penalty: 'Repeat penalty',
		temperature: 'Temperature',
		top_k: 'Top K',
		top_p: 'Top P'
	};
	const LEVELS = [
		{ key: 'low', label: 'Low' },
		{ key: 'medium', label: 'Medium' },
		{ key: 'high', label: 'High' },
		{ key: 'max', label: 'Max' }
	] as const;
	const PRESETS = [
		{ key: 'thinking', label: 'Thinking', when: 'an explicit reasoning level' },
		{ key: 'instruct', label: 'Instruct', when: 'reasoning Off' }
	] as const;

	let active = $derived(
		activePresetLine(reasoning.currentEffort, reasoning.modelSupportsThinking, presets)
	);
	let thinkingValues = $derived(presetValues('thinking', presets).values);

	type Target = 'thinking' | 'instruct' | 'low' | 'medium' | 'high' | 'max';

	/** Apply an edit to the draft; an invalid value (e.g. a non-integer top_k) is not taken. */
	function edit(target: Target, field: PresetField, input: HTMLInputElement) {
		const raw = input.value.trim();

		if (raw === '') {
			presets = withPresetField(presets, target, field, undefined);

			return;
		}

		const v = sanitizePresetValue(field, Number(raw));

		if (v === undefined) {
			input.setCustomValidity(field === 'top_k' ? 'Top K must be a whole number' : 'Not a number');
			input.reportValidity();

			return;
		}

		input.setCustomValidity('');
		presets = withPresetField(presets, target, field, v);
	}
</script>

<div class="space-y-4 border-t border-border/40 pt-6" data-sampling-presets>
	<div>
		<h3 class="text-sm font-medium">Presets by reasoning state</h3>

		<p class="mt-1 text-xs text-muted-foreground">
			Sent with an explicit reasoning level (Thinking) or Off (Instruct) on a model that supports
			thinking, for the sampling fields left empty above - your own values there always win. Default
			sends none: the server's sampling applies.
		</p>

		<p class="mt-1 text-xs" data-active-preset>Active in this chat: {active}</p>
	</div>

	{#each PRESETS as preset (preset.key)}
		{@const p = presetValues(preset.key, presets)}

		<div class="space-y-2">
			<div class="flex items-center justify-between gap-2">
				<span class="text-sm">
					{preset.label}
					<span class="text-xs text-muted-foreground">
						({preset.when}{p.custom ? ', edited' : ', model card'})
					</span>
				</span>

				<Button
					disabled={!p.custom}
					onclick={() => (presets = withoutPreset(presets, preset.key))}
					size="sm"
					variant="outline"
				>
					Reset to model card
				</Button>
			</div>

			<div class="grid grid-cols-2 gap-2 sm:grid-cols-3">
				{#each PRESET_FIELDS as field (field)}
					<label class="grid gap-1 text-xs text-muted-foreground">
						{FIELD_LABELS[field]}

						<Input
							inputmode="decimal"
							onchange={(e) => edit(preset.key, field, e.currentTarget)}
							placeholder={String(SAMPLING_PRESETS[preset.key][field])}
							value={String(p.values[field])}
						/>
					</label>
				{/each}
			</div>
		</div>
	{/each}

	<details class="rounded-md border border-border/40 p-2">
		<summary class="cursor-pointer text-sm">Per-level overrides</summary>

		<p class="mt-2 text-xs text-muted-foreground">
			Optional, per reasoning level: an empty field inherits the Thinking preset; a filled one
			overrides it for that level only.
		</p>

		{#each LEVELS as level (level.key)}
			{@const o = presets.levels?.[level.key] ?? {}}

			<div class="mt-3 space-y-1">
				<span class="text-xs font-medium">{level.label}</span>

				<div class="grid grid-cols-2 gap-2 sm:grid-cols-3">
					{#each PRESET_FIELDS as field (field)}
						<label class="grid gap-1 text-xs text-muted-foreground">
							{FIELD_LABELS[field]}

							<Input
								inputmode="decimal"
								onchange={(e) => edit(level.key, field, e.currentTarget)}
								placeholder={String(thinkingValues[field])}
								value={o[field] !== undefined ? String(o[field]) : ''}
							/>
						</label>
					{/each}
				</div>
			</div>
		{/each}
	</details>
</div>
