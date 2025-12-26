/** @type {import('next').NextConfig} */
const r2PublicHostname = process.env.NEXT_PUBLIC_R2_PUBLIC_HOSTNAME;

const nextConfig = {
  reactStrictMode: true,
  images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "9000",
      },
      {
        protocol: "https",
        hostname: "*.r2.cloudflarestorage.com",
      },
      ...(r2PublicHostname
        ? [
            {
              protocol: "https",
              hostname: r2PublicHostname,
            },
          ]
        : []),
    ],
  },
};

module.exports = nextConfig;






