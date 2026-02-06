<script setup lang="ts">
import { Link, usePage } from '@inertiajs/vue3';
import { AlertCircle, Leaf } from 'lucide-vue-next';
import { computed } from 'vue';

const flash = computed(
    () => (usePage().props as unknown as { flash: { error?: { title: string; description: string } } }).flash,
);
</script>

<template>
    <div>
        <header class="sticky top-0 z-10 w-full border-b border-border/60 bg-card/80 backdrop-blur-sm">
            <div class="mx-auto flex h-14 max-w-4xl items-center justify-between px-4">
                <Link href="/" class="flex items-center gap-2 transition-opacity hover:opacity-80">
                    <div class="flex h-8 w-8 items-center justify-center rounded bg-primary">
                        <Leaf class="h-5 w-5 text-primary-foreground" />
                    </div>
                    <span class="font-serif text-lg font-semibold tracking-wider text-foreground">다육도감</span>
                </Link>
            </div>
        </header>
        <main class="mx-auto max-w-4xl px-4 py-8 pb-16">
            <div v-if="flash.error" class="mb-6 flex gap-2 rounded-lg bg-destructive/10 p-3 text-destructive" role="alert">
                <AlertCircle class="mt-0.5 h-4 w-4 shrink-0" />
                <div>
                    <p class="font-medium">{{ flash.error.title }}</p>
                    <p class="text-sm">{{ flash.error.description }}</p>
                </div>
            </div>
            <slot />
        </main>
    </div>
</template>
