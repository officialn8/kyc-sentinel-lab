"use client";

import * as React from "react";
import { motion, useMotionValue, useTransform, useSpring } from "framer-motion";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";
import { buttonVariants as motionButtonVariants, springs } from "@/lib/animations";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-lg text-sm font-medium ring-offset-background transition-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 relative overflow-hidden",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground shadow-elevation-2",
        destructive: "bg-destructive text-destructive-foreground shadow-elevation-2",
        outline: "border border-input bg-background shadow-sm",
        secondary: "bg-secondary text-secondary-foreground shadow-elevation-1",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
        danger: "bg-danger text-danger-foreground shadow-elevation-2",
        success: "bg-success text-success-foreground shadow-elevation-2",
      },
      size: {
        default: "h-10 px-5 py-2",
        sm: "h-9 px-4 text-xs",
        lg: "h-12 px-8 text-base",
        xl: "h-14 px-10 text-lg",
        icon: "h-10 w-10",
      },
      glow: {
        true: "",
        false: "",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
      glow: false,
    },
    compoundVariants: [
      {
        variant: "default",
        glow: true,
        className: "shadow-glow-primary",
      },
      {
        variant: "danger",
        glow: true,
        className: "shadow-glow-danger",
      },
      {
        variant: "success",
        glow: true,
        className: "shadow-glow-success",
      },
    ],
  }
);

export interface EnhancedButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
  magnetic?: boolean;
  ripple?: boolean;
  haptic?: boolean;
  soundEffect?: string;
}

interface RippleData {
  x: number;
  y: number;
  size: number;
  id: number;
}

const EnhancedButton = React.forwardRef<HTMLButtonElement, EnhancedButtonProps>(
  (
    {
      className,
      variant,
      size,
      glow,
      asChild = false,
      magnetic = true,
      ripple = true,
      haptic = true,
      soundEffect = "click-soft",
      disabled,
      onClick,
      onMouseMove,
      onMouseLeave,
      children,
      ...props
    },
    ref
  ) => {
    const Comp = asChild ? Slot : "button";
    const buttonRef = React.useRef<HTMLButtonElement>(null);
    const [ripples, setRipples] = React.useState<RippleData[]>([]);

    // Magnetic hover effect
    const x = useMotionValue(0);
    const y = useMotionValue(0);
    const springX = useSpring(x, springs.snappy);
    const springY = useSpring(y, springs.snappy);

    // Handle mouse move for magnetic effect
    const handleMouseMove = React.useCallback(
      (e: React.MouseEvent<HTMLButtonElement>) => {
        if (!magnetic || disabled) return;

        const rect = buttonRef.current?.getBoundingClientRect();
        if (!rect) return;

        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const distanceX = (e.clientX - centerX) * 0.1;
        const distanceY = (e.clientY - centerY) * 0.1;

        x.set(distanceX);
        y.set(distanceY);

        onMouseMove?.(e);
      },
      [magnetic, disabled, x, y, onMouseMove]
    );

    // Handle mouse leave
    const handleMouseLeave = React.useCallback(
      (e: React.MouseEvent<HTMLButtonElement>) => {
        x.set(0);
        y.set(0);
        onMouseLeave?.(e);
      },
      [x, y, onMouseLeave]
    );

    // Handle click with ripple effect
    const handleClick = React.useCallback(
      (e: React.MouseEvent<HTMLButtonElement>) => {
        if (ripple && !disabled) {
          const rect = buttonRef.current?.getBoundingClientRect();
          if (rect) {
            const size = Math.max(rect.width, rect.height) * 2;
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;

            const newRipple = {
              x,
              y,
              size,
              id: Date.now(),
            };

            setRipples((prev) => [...prev, newRipple]);

            // Remove ripple after animation
            setTimeout(() => {
              setRipples((prev) => prev.filter((r) => r.id !== newRipple.id));
            }, 600);
          }
        }

        // Haptic feedback (if supported)
        if (haptic && "vibrate" in navigator) {
          navigator.vibrate(10);
        }

        // Sound effect attribute for future implementation
        if (soundEffect) {
          (e.currentTarget as HTMLElement).setAttribute("data-sound", soundEffect);
        }

        onClick?.(e);
      },
      [ripple, haptic, soundEffect, disabled, onClick]
    );

    // When using asChild, we can't wrap with motion.div as it breaks Slot
    if (asChild) {
      return (
        <Comp
          ref={(node: HTMLButtonElement | null) => {
            (buttonRef as React.MutableRefObject<HTMLButtonElement | null>).current = node;
            if (ref) {
              if (typeof ref === "function") {
                ref(node);
              } else {
                (ref as React.MutableRefObject<HTMLButtonElement | null>).current = node;
              }
            }
          }}
          className={cn(buttonVariants({ variant, size, glow, className }))}
          disabled={disabled}
          onClick={handleClick}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          {...props}
        >
          {children}
        </Comp>
      );
    }

    return (
      <motion.div
        className="inline-flex"
        style={{ x: springX, y: springY }}
        whileHover={!disabled ? "hover" : undefined}
        whileTap={!disabled ? "tap" : undefined}
        initial="idle"
        variants={motionButtonVariants}
      >
        <Comp
          ref={(node: HTMLButtonElement | null) => {
            (buttonRef as React.MutableRefObject<HTMLButtonElement | null>).current = node;
            if (ref) {
              if (typeof ref === "function") {
                ref(node);
              } else {
                (ref as React.MutableRefObject<HTMLButtonElement | null>).current = node;
              }
            }
          }}
          className={cn(buttonVariants({ variant, size, glow, className }))}
          disabled={disabled}
          onClick={handleClick}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          {...props}
        >
          {/* Ripple effects container */}
          {ripple && (
            <span className="absolute inset-0 overflow-hidden rounded-[inherit]">
              {ripples.map((ripple) => (
                <motion.span
                  key={ripple.id}
                  className="absolute rounded-full bg-white/30"
                  style={{
                    left: ripple.x,
                    top: ripple.y,
                    width: ripple.size,
                    height: ripple.size,
                  }}
                  initial={{ scale: 0, opacity: 1 }}
                  animate={{ scale: 1, opacity: 0 }}
                  transition={{
                    duration: 0.6,
                    ease: [0.4, 0, 0.2, 1],
                  }}
                />
              ))}
            </span>
          )}

          {/* Gradient shine effect on hover */}
          <motion.span
            className="absolute inset-0 rounded-[inherit] opacity-0"
            style={{
              background:
                "linear-gradient(105deg, transparent 40%, rgba(255, 255, 255, 0.7) 50%, transparent 60%)",
              mixBlendMode: "overlay",
            }}
            initial={{ x: "-100%" }}
            whileHover={{
              opacity: 1,
              x: "100%",
              transition: { duration: 0.5, ease: "easeInOut" },
            }}
          />

          {/* Content with proper z-index */}
          <span className="relative z-10">{children}</span>

          {/* Focus ring enhancement */}
          <motion.span
            className="absolute inset-0 rounded-[inherit] ring-2 ring-primary ring-offset-2 ring-offset-background opacity-0"
            initial={false}
            whileFocus={{ opacity: 1 }}
            transition={springs.snappy}
          />
        </Comp>
      </motion.div>
    );
  }
);

EnhancedButton.displayName = "EnhancedButton";

export { EnhancedButton, buttonVariants };