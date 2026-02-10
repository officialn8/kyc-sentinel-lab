import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

const isPublicRoute = createRouteMatcher([
  "/sign-in(.*)",
  "/sign-up(.*)",
  "/favicon.ico",
]);
const isOrganizationSetupRoute = createRouteMatcher(["/organization/create(.*)"]);

export default clerkMiddleware(async (auth, req) => {
  if (isPublicRoute(req)) {
    return;
  }

  await auth.protect();

  const { userId, orgId } = await auth();
  if (
    userId &&
    !orgId &&
    !isOrganizationSetupRoute(req) &&
    !req.nextUrl.pathname.startsWith("/api")
  ) {
    const createOrgUrl = new URL("/organization/create", req.url);
    const returnBackUrl = `${req.nextUrl.pathname}${req.nextUrl.search}`;
    if (returnBackUrl && returnBackUrl !== "/organization/create") {
      createOrgUrl.searchParams.set("redirect_url", returnBackUrl);
    }
    return NextResponse.redirect(createOrgUrl);
  }
});

export const config = {
  matcher: [
    // Skip Next internals/static files while protecting app routes.
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpg|jpeg|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    // Always run for API routes.
    "/(api|trpc)(.*)",
  ],
};
