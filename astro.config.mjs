// @ts-check
import { defineConfig } from 'astro/config';

import react from '@astrojs/react';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
  site: 'https://under-formation.com',
  // `/architecture/` is internal notes about the codebase: kept out of the sitemap.
  integrations: [react(), mdx(), sitemap({ filter: (page) => !page.includes('/architecture') })],
  markdown: {
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
    },
  },
});
