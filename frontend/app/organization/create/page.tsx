import { CreateOrganization } from "@clerk/nextjs";
import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";

type CreateOrganizationPageProps = {
  searchParams?: Promise<{ redirect_url?: string }>;
};

function getSafeRedirectUrl(value: string | undefined): string {
  if (!value || !value.startsWith("/") || value.startsWith("//")) {
    return "/";
  }
  return value;
}

export default async function CreateOrganizationPage({
  searchParams,
}: CreateOrganizationPageProps) {
  const { userId, orgId } = await auth();
  if (!userId) {
    redirect("/sign-in");
  }

  const resolvedParams = searchParams ? await searchParams : undefined;
  const redirectUrl = getSafeRedirectUrl(resolvedParams?.redirect_url);

  if (orgId) {
    redirect(redirectUrl);
  }

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col items-center gap-4 px-4 py-6 md:py-10">
      <h1 className="text-2xl font-semibold tracking-tight">Create your organization</h1>
      <p className="max-w-2xl text-center text-sm text-muted-foreground">
        KYC Sentinel is tenant-scoped. Create an organization, invite your team, and
        continue into your isolated workspace.
      </p>
      <div className="w-full max-w-lg">
        <CreateOrganization afterCreateOrganizationUrl={redirectUrl} />
      </div>
    </div>
  );
}
