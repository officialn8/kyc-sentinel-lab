import { Variants } from "framer-motion";

// Spring physics presets for different animation feels
export const springs = {
  // Snappy, responsive interactions (buttons, toggles)
  snappy: { type: "spring", stiffness: 300, damping: 30 },

  // Smooth, natural transitions (page transitions, modals)
  smooth: { type: "spring", stiffness: 100, damping: 20 },

  // Bouncy, playful feedback (success states, notifications)
  bouncy: { type: "spring", stiffness: 400, damping: 10 },

  // Heavy, dramatic movement (large panels, important reveals)
  heavy: { type: "spring", stiffness: 200, damping: 40 },

  // Ultra responsive, no bounce (micro-interactions)
  quick: { type: "tween", duration: 0.15, ease: [0.4, 0, 0.2, 1] },

  // Slow, graceful (background animations)
  gentle: { type: "spring", stiffness: 50, damping: 15 },
} as const;

// Transition presets for different contexts
export const transitions = {
  // Hover states
  hover: { duration: 0.15, ease: [0.4, 0, 0.2, 1] },

  // Button/tap interactions
  tap: { duration: 0.1, ease: [0.4, 0, 1, 1] },

  // Page/route transitions
  page: { duration: 0.3, ease: [0.8, 0, 0.2, 1] },

  // Stagger children animations
  stagger: { staggerChildren: 0.05, delayChildren: 0.02 },

  // Stagger with spring
  staggerSpring: { staggerChildren: 0.05, delayChildren: 0.02, ...springs.smooth },
} as const;

// Reusable animation variants
export const variants = {
  // Fade animations
  fadeIn: {
    initial: { opacity: 0 },
    animate: { opacity: 1, transition: transitions.page },
    exit: { opacity: 0, transition: transitions.page },
  } as Variants,

  // Scale + fade (for modals, cards)
  scaleIn: {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1, transition: springs.smooth },
    exit: { opacity: 0, scale: 0.95, transition: springs.smooth },
  } as Variants,

  // Slide animations
  slideUp: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0, transition: springs.smooth },
    exit: { opacity: 0, y: 20, transition: springs.smooth },
  } as Variants,

  slideDown: {
    initial: { opacity: 0, y: -20 },
    animate: { opacity: 1, y: 0, transition: springs.smooth },
    exit: { opacity: 0, y: -20, transition: springs.smooth },
  } as Variants,

  slideRight: {
    initial: { opacity: 0, x: -20 },
    animate: { opacity: 1, x: 0, transition: springs.smooth },
    exit: { opacity: 0, x: -20, transition: springs.smooth },
  } as Variants,

  slideLeft: {
    initial: { opacity: 0, x: 20 },
    animate: { opacity: 1, x: 0, transition: springs.smooth },
    exit: { opacity: 0, x: 20, transition: springs.smooth },
  } as Variants,

  // 3D perspective animations
  perspective: {
    initial: { opacity: 0, rotateX: 25, scale: 0.9 },
    animate: { opacity: 1, rotateX: 0, scale: 1, transition: springs.smooth },
    exit: { opacity: 0, rotateX: -25, scale: 0.9, transition: springs.smooth },
  } as Variants,

  // Container variants for staggered children
  container: {
    initial: { opacity: 0 },
    animate: {
      opacity: 1,
      transition: transitions.staggerSpring,
    },
    exit: { opacity: 0 },
  } as Variants,
};

// Button animation variants with multi-layer feedback
export const buttonVariants = {
  idle: { scale: 1, y: 0 },
  hover: {
    scale: 1.02,
    y: -2,
    transition: springs.snappy,
  },
  tap: {
    scale: 0.95,
    y: 1,
    transition: springs.quick,
  },
  disabled: {
    scale: 1,
    opacity: 0.5,
  },
} as const;

// Card hover animation with 3D lift
export const cardHoverVariants = {
  idle: {
    y: 0,
    scale: 1,
    rotateX: 0,
    boxShadow: "0 8px 16px rgba(0, 0, 0, 0.12), 0 4px 8px rgba(0, 0, 0, 0.08)",
  },
  hover: {
    y: -8,
    scale: 1.01,
    rotateX: 5,
    boxShadow: "0 16px 32px rgba(0, 0, 0, 0.16), 0 8px 16px rgba(0, 0, 0, 0.08)",
    transition: springs.snappy,
  },
  tap: {
    scale: 0.98,
    transition: springs.quick,
  },
} as const;

// List item animations for staggered appearance
export const listItemVariants = {
  initial: { opacity: 0, y: 20, scale: 0.98 },
  animate: (index: number) => ({
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      ...springs.smooth,
      delay: index * 0.05,
    },
  }),
  exit: { opacity: 0, y: -20, scale: 0.98 },
} as const;

// Progress ring animation for loading states
export const progressRingVariants = {
  initial: { rotate: 0 },
  animate: {
    rotate: 360,
    transition: {
      duration: 1.5,
      ease: "linear",
      repeat: Infinity,
    },
  },
} as const;

// Shake animation for errors
export const shakeVariants = {
  initial: { x: 0 },
  shake: {
    x: [-10, 10, -10, 10, -5, 5, -2, 2, 0],
    transition: {
      duration: 0.5,
      ease: "easeInOut",
    },
  },
} as const;

// Glow pulse for attention
export const glowPulseVariants = {
  initial: {
    boxShadow: "0 0 20px rgba(239, 68, 68, 0)",
  },
  animate: {
    boxShadow: [
      "0 0 20px rgba(239, 68, 68, 0)",
      "0 0 40px rgba(239, 68, 68, 0.6)",
      "0 0 20px rgba(239, 68, 68, 0)",
    ],
    transition: {
      duration: 2,
      ease: "easeInOut",
      repeat: Infinity,
    },
  },
} as const;

// Floating animation for background elements
export const floatVariants = {
  initial: { y: 0 },
  animate: {
    y: [-10, 10, -10],
    transition: {
      duration: 6,
      ease: "easeInOut",
      repeat: Infinity,
    },
  },
} as const;

// Page transition variants
export const pageTransitionVariants = {
  initial: { opacity: 0, scale: 0.98 },
  animate: {
    opacity: 1,
    scale: 1,
    transition: {
      ...springs.smooth,
      staggerChildren: 0.1,
    },
  },
  exit: {
    opacity: 0,
    scale: 0.98,
    transition: springs.smooth,
  },
} as const;

// Success checkmark draw animation
export const checkmarkVariants = {
  initial: { pathLength: 0, opacity: 0 },
  animate: {
    pathLength: 1,
    opacity: 1,
    transition: {
      pathLength: { delay: 0.2, ...springs.bouncy },
      opacity: { delay: 0.2, duration: 0.1 },
    },
  },
} as const;

// Utility function for magnetic hover effect
export const magneticHover = (strength: number = 20) => ({
  whileHover: (e: MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    return {
      x: x / rect.width * strength,
      y: y / rect.height * strength,
    };
  },
});

// Scroll-triggered animation helpers
export const scrollTriggerVariants = {
  offscreen: {
    y: 50,
    opacity: 0,
    scale: 0.98,
  },
  onscreen: {
    y: 0,
    opacity: 1,
    scale: 1,
    transition: {
      ...springs.smooth,
      staggerChildren: 0.05,
    },
  },
} as const;

// Parallax scroll variants
export const parallaxVariants = (speed: number = 0.5) => ({
  initial: { y: 0 },
  animate: {
    y: 0,
    transition: springs.smooth,
  },
  // Use with scroll progress
  scroll: (progress: number) => ({
    y: progress * speed * 100,
  }),
});

// Export all as a single object for convenience
export const animations = {
  springs,
  transitions,
  variants,
  buttonVariants,
  cardHoverVariants,
  listItemVariants,
  progressRingVariants,
  shakeVariants,
  glowPulseVariants,
  floatVariants,
  pageTransitionVariants,
  checkmarkVariants,
  scrollTriggerVariants,
  parallaxVariants,
  magneticHover,
} as const;

// Type exports
export type SpringPreset = keyof typeof springs;
export type TransitionPreset = keyof typeof transitions;
export type AnimationVariant = keyof typeof variants;