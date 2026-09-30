// The site is served from a subpath (`base` in astro.config.mjs), so every internal link and
// public asset path goes through this. Write paths as if the site lived at `/`.
const base = import.meta.env.BASE_URL.replace(/\/$/, '');

export const url = (path: string) => base + path;
