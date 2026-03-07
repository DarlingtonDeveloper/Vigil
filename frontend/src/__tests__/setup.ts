import "@testing-library/jest-dom/vitest";

// Mock scrollIntoView (not implemented in jsdom)
Element.prototype.scrollIntoView = vi.fn();

// Mock next/dynamic for KnowledgeGraph
vi.mock("next/dynamic", () => ({
  __esModule: true,
  default: (loader: () => Promise<any>) => {
    // Return a simple component that renders nothing for SSR-disabled dynamic imports
    const Component = (props: any) => null;
    Component.displayName = "DynamicComponent";
    return Component;
  },
}));
