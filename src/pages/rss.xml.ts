import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { SITE_TITLE, SITE_DESCRIPTION } from '../consts';
import { getPosts, getVisualizations } from '../lib/content';
import { url } from '../lib/url';

export async function GET(context: APIContext) {
  const posts = (await getPosts()).map((p) => ({
    title: p.data.title,
    description: p.data.description,
    pubDate: p.data.pubDate,
    link: url(`/blog/${p.id}/`),
  }));
  const viz = (await getVisualizations()).map((v) => ({
    title: `Visualized: ${v.data.title}`,
    description: v.data.description,
    pubDate: v.data.pubDate,
    link: url(`/visualize/${v.id}/`),
  }));

  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: new URL(url('/'), context.site),
    items: [...posts, ...viz].sort((a, b) => b.pubDate.valueOf() - a.pubDate.valueOf()),
  });
}
