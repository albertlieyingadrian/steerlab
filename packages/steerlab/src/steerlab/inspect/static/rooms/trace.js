/* The Trace room is the original layer explorer (index.html's #room-trace and inspect.js), shown and hidden by the
   shell directly, so this module only exists to satisfy the room contract for tools that import every room. */
export function mount() { window.dispatchEvent(new Event("steerlab:trace")); }
export function unmount() {}
