// @ts-check
import { defineConfig } from 'astro/config';

import react from '@astrojs/react';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { satteri } from '@astrojs/markdown-satteri';
import { SHOW_BLOG } from './src/consts';

// GitHub Pages serves the repo under /under-formation/. Components prefix links with url() from
// src/lib/url.ts; this does the same for root-relative links written in Markdown/MDX.
const BASE = '/under-formation';
/** @type {import('satteri').HastPluginDefinition} */
const prefixMarkdownLinks = {
  name: 'prefix-markdown-links',
  element: {
    filter: ['a'],
    visit(node, ctx) {
      const href = node.properties.href;
      if (typeof href === 'string' && href.startsWith('/') && !href.startsWith('//')) {
        ctx.setProperty(node, 'href', BASE + href);
      }
    },
  },
};

// https://astro.build/config
export default defineConfig({
  site: 'https://thakurabhishek7283.github.io',
  base: BASE,
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
    processor: satteri({ hastPlugins: [prefixMarkdownLinks] }),
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
    },
  },
});
