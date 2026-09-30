/** Named view links beside the control rail. The list lives for this viewer session. */
import type { ViewBookmark } from '../../core/app/bookmark-state';
import { downloadBlob } from '../recording-panel/screenshot-exporter';

/** Live viewer operations supplied by the app; the panel owns no scene state. */
export interface BookmarksPopoverContext {
  capture(): ViewBookmark;
  restore(bookmark: ViewBookmark): Promise<void>;
  baseUrl(): string;
  buildUrl(base: string, bookmark: ViewBookmark): string;
  copy(url: string): Promise<void>;
}

/** A captured view and the link copied when it was added. */
export interface NamedBookmark {
  name: string;
  url: string;
  state: ViewBookmark;
}

/** Build a panel over a persistent list owned by the rail item closure. */
export function buildBookmarksPopover(
  host: HTMLElement,
  ctx: BookmarksPopoverContext,
  bookmarks: NamedBookmark[]
): () => void {
  const root = document.createElement('div');
  root.className = 'luxar-bookmarks';
  const heading = document.createElement('strong');
  heading.textContent = 'Bookmarks';
  const form = document.createElement('form');
  form.className = 'luxar-bookmarks__form';
  const name = document.createElement('input');
  name.type = 'text';
  name.placeholder = 'View name';
  name.setAttribute('aria-label', 'Bookmark name');
  name.maxLength = 100;
  const add = document.createElement('button');
  add.type = 'submit';
  add.textContent = 'Add bookmark';
  form.append(name, add);
  const list = document.createElement('ul');
  list.setAttribute('aria-label', 'Saved views');
  const actions = document.createElement('div');
  actions.className = 'luxar-bookmarks__actions';
  const clear = document.createElement('button');
  clear.type = 'button';
  clear.textContent = 'Clear';
  const save = document.createElement('button');
  save.type = 'button';
  save.textContent = 'Save';
  actions.append(clear, save);
  const status = document.createElement('div');
  status.className = 'luxar-bookmarks__status';
  status.setAttribute('role', 'status');

  const render = (): void => {
    list.replaceChildren();
    for (const bookmark of bookmarks) {
      const item = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = bookmark.name;
      button.title = bookmark.url;
      button.addEventListener('click', () => {
        void ctx.restore(bookmark.state).then(
          () => {
            status.textContent = `Opened ${bookmark.name}`;
          },
          () => {
            status.textContent = 'Could not open bookmark';
          }
        );
      });
      item.appendChild(button);
      list.appendChild(item);
    }
    clear.disabled = save.disabled = bookmarks.length === 0;
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    try {
      const state = ctx.capture();
      const label = name.value.trim().replace(/[\r\n\t]+/g, ' ') || `View ${bookmarks.length + 1}`;
      const url = ctx.buildUrl(ctx.baseUrl(), state);
      bookmarks.push({ name: label, url, state });
      name.value = '';
      render();
      void ctx.copy(url).then(
        () => {
          status.textContent = 'Link copied';
        },
        () => {
          status.textContent = 'Bookmark added; clipboard unavailable';
        }
      );
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : 'Could not add bookmark';
    }
  });
  clear.addEventListener('click', () => {
    bookmarks.length = 0;
    render();
    status.textContent = 'Bookmarks cleared';
  });
  save.addEventListener('click', () => {
    const lines = bookmarks.map(({ name: label, url }) => `${label}\t${url}`);
    downloadBlob(
      new Blob([lines.join('\n') + '\n'], { type: 'text/plain' }),
      'luxar-bookmarks.txt'
    );
    status.textContent = 'Bookmarks saved';
  });
  root.append(heading, form, list, actions, status);
  host.appendChild(root);
  render();
  return () => {};
}
