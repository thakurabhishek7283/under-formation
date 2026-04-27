import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const blog = defineCollection({
  loader: glob({ base: './src/content/blog', pattern: '**/*.{md,mdx}' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});

const projects = defineCollection({
  loader: glob({ base: './src/content/projects', pattern: '**/*.{md,mdx}' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    tech: z.array(z.string()).default([]),
    status: z.enum(['live', 'wip', 'archived']).default('live'),
    featured: z.boolean().default(false),
    repo: z.url().optional(),
    // A deployed URL. If `embed` is true it is shown live in an iframe on the project page.
    demo: z
      .object({
        url: z.url(),
        embed: z.boolean().default(true),
        height: z.number().default(600),
      })
      .optional(),
    draft: z.boolean().default(false),
  }),
});

const visualizations = defineCollection({
  loader: glob({ base: './src/content/visualizations', pattern: '**/*.{md,mdx}' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    // Groups cards on /visualize, e.g. "Probabilistic data structures", "Graphs", "Sorting".
    category: z.string(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});

export const collections = { blog, projects, visualizations };
