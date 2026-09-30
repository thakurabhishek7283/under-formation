// @ts-check
import { defineConfig } from 'astro/config';

import react from '@astrojs/react';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { SHOW_BLOG } from './src/consts';

// https://astro.build/config
export default defineConfig({
  site: 'https://thakurabhishek7283.github.io',
  // `/architecture/` is internal notes about the codebase: kept out of the sitemap, as is the
  // blog while it's hidden.
  integrations: [
    react(),
    mdx(),
    sitemap({
      filter: (page) => !page.includes('/architecture') && (SHOW_BLOG || !page.includes('/blog')),
    }),
  ],
  markdown: {
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
    },
  },
});
