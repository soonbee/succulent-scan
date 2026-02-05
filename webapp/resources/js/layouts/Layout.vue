<script setup lang="ts">
import { Link } from '@inertiajs/vue3';
import { Leaf } from 'lucide-vue-next';
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
            <slot />
        </main>
    </div>
</template>
