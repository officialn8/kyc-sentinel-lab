"use client";

import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

// Success checkmark animation
interface SuccessAnimationProps {
  show: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
  onAnimationComplete?: () => void;
}

export function SuccessAnimation({
  show,
  size = "md",
  className,
  onAnimationComplete,
}: SuccessAnimationProps) {
  const sizes = {
    sm: { container: "w-12 h-12", icon: "h-6 w-6" },
    md: { container: "w-16 h-16", icon: "h-8 w-8" },
    lg: { container: "w-24 h-24", icon: "h-12 w-12" },
  };

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0, opacity: 0 }}
          transition={{
            type: "spring",
            stiffness: 300,
            damping: 20,
          }}
          onAnimationComplete={onAnimationComplete}
          className={cn(
            "flex items-center justify-center rounded-full bg-success/20",
            sizes[size].container,
            className
          )}
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{
              type: "spring",
              stiffness: 400,
              damping: 15,
              delay: 0.1,
            }}
          >
            <CheckCircle2 className={cn("text-success", sizes[size].icon)} />
          </motion.div>
          {/* Ripple effect */}
          <motion.div
            initial={{ scale: 0.8, opacity: 1 }}
            animate={{ scale: 2, opacity: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className={cn(
              "absolute rounded-full border-2 border-success/50",
              sizes[size].container
            )}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// Progress ring animation
interface ProgressRingProps {
  progress: number; // 0-100
  size?: number;
  strokeWidth?: number;
  className?: string;
  showPercentage?: boolean;
  indeterminate?: boolean;
}

export function ProgressRing({
  progress,
  size = 48,
  strokeWidth = 4,
  className,
  showPercentage = true,
  indeterminate = false,
}: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  return (
    <div className={cn("relative inline-flex items-center justify-center", className)}>
      <svg
        width={size}
        height={size}
        className={cn(indeterminate && "animate-spin")}
        style={{ transform: "rotate(-90deg)" }}
      >
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-muted/30"
        />
        {/* Progress circle */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          className="text-primary"
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: indeterminate ? circumference * 0.75 : strokeDashoffset }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          style={{
            strokeDasharray: circumference,
          }}
        />
      </svg>
      {showPercentage && !indeterminate && (
        <motion.span
          className="absolute text-xs font-medium"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          key={Math.round(progress)}
        >
          {Math.round(progress)}%
        </motion.span>
      )}
    </div>
  );
}

// Animated number counter
interface AnimatedNumberProps {
  value: number;
  duration?: number;
  className?: string;
  formatFn?: (value: number) => string;
}

export function AnimatedNumber({
  value,
  duration = 0.5,
  className,
  formatFn = (v) => Math.round(v).toString(),
}: AnimatedNumberProps) {
  return (
    <motion.span
      className={className}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      key={value}
      transition={{ duration: duration / 2 }}
    >
      {formatFn(value)}
    </motion.span>
  );
}

// List item animation wrapper
interface AnimatedListItemProps {
  children: React.ReactNode;
  index?: number;
  className?: string;
}

export function AnimatedListItem({
  children,
  index = 0,
  className,
}: AnimatedListItemProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{
        duration: 0.2,
        delay: index * 0.05,
        ease: "easeOut",
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// Fade in animation wrapper
interface FadeInProps {
  children: React.ReactNode;
  delay?: number;
  duration?: number;
  className?: string;
}

export function FadeIn({
  children,
  delay = 0,
  duration = 0.3,
  className,
}: FadeInProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration, delay }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// Scale in animation wrapper (good for modals/cards)
interface ScaleInProps {
  children: React.ReactNode;
  className?: string;
}

export function ScaleIn({ children, className }: ScaleInProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{
        type: "spring",
        stiffness: 300,
        damping: 25,
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// Pulse animation for attention
interface PulseProps {
  children: React.ReactNode;
  active?: boolean;
  className?: string;
}

export function Pulse({ children, active = true, className }: PulseProps) {
  return (
    <motion.div
      animate={active ? { scale: [1, 1.05, 1] } : {}}
      transition={{
        duration: 2,
        repeat: Infinity,
        repeatType: "loop",
        ease: "easeInOut",
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// Processing spinner with status text
interface ProcessingSpinnerProps {
  status?: string;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function ProcessingSpinner({
  status = "Processing...",
  size = "md",
  className,
}: ProcessingSpinnerProps) {
  const sizes = {
    sm: { ring: 32, stroke: 3 },
    md: { ring: 48, stroke: 4 },
    lg: { ring: 64, stroke: 5 },
  };

  return (
    <div className={cn("flex flex-col items-center gap-3", className)}>
      <ProgressRing
        progress={0}
        size={sizes[size].ring}
        strokeWidth={sizes[size].stroke}
        indeterminate
        showPercentage={false}
      />
      <motion.p
        className="text-sm text-muted-foreground"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        key={status}
      >
        {status}
      </motion.p>
    </div>
  );
}
