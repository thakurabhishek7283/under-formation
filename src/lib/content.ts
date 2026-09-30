import { getCollection } from 'astro:content';
import { SHOW_BLOG } from '../consts';

// Drafts are visible in `npm run dev` but excluded from production builds.
const visible = ({ data }: { data: { draft: boolean } }) => import.meta.env.DEV || !data.draft;

export async function getPosts() {
  if (!SHOW_BLOG) return [];
  const posts = await getCollection('blog', visible);
  return posts.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}

export async function getProjects() {
  const projects = await getCollection('projects', visible);
  return projects.sort(
    (a, b) => Number(b.data.featured) - Number(a.data.featured) || b.data.date.valueOf() - a.data.date.valueOf(),
  );
}

export async function getVisualizations() {
  const viz = await getCollection('visualizations', visible);
  return viz.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}

export const STATUS_LABEL = { live: 'Live demo', wip: 'In progress', archived: 'Archived' } as const;
