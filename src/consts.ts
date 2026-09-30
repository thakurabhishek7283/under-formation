// Site-wide settings. Edit these to make the site yours.

export const SITE_TITLE = 'Under Formation';
export const SITE_DESCRIPTION =
  'Abhishek Thakur, software engineer building .NET backends, search and RAG systems. Projects, interactive visualizations of things I have learned, and Loose Threads.';

export const AUTHOR = {
  name: 'Abhishek Thakur',
  tagline: 'Software engineer interested in machines that can think, move, and learn',
  email: 'thakurabhishek7283@gmail.com',
  role: 'Software Engineer at Eurofins IT Delivery Center',
};

export const SOCIALS: { label: string; href: string }[] = [
  { label: 'GitHub', href: 'https://github.com/thakurabhishek7283' },
  { label: 'LinkedIn', href: 'https://www.linkedin.com/in/abhishek-thakur-072a981b6/' },
  { label: 'LeetCode', href: 'https://leetcode.com/u/abhishekthakur7283/' },
];

// The blog is hidden until there are posts worth publishing. Flip to true to bring back the nav
// link, the home page section, the post pages and the posts in the RSS feed.
export const SHOW_BLOG = false;

export const NAV_LINKS = [
  { label: 'About', href: '/about/' },
  { label: 'Projects', href: '/projects/' },
  { label: 'Visualize', href: '/visualize/' },
  ...(SHOW_BLOG ? [{ label: 'Blog', href: '/blog/' }] : []),
];
