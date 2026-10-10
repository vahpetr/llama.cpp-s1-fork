<script lang="ts">
	// W2.1: hands a tool block's chip data to its header (ToolCallBlock) through a context
	import { setToolChipContext, type ToolChipContext } from '$lib/contexts/tool-chip.context';
	import type { Snippet } from 'svelte';

	interface Props extends ToolChipContext {
		children: Snippet;
	}

	let { children, executingSince, record, visible }: Props = $props();

	setToolChipContext(() => ({ executingSince, record, visible }));
</script>

{@render children()}
