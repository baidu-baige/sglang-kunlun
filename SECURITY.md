<!--
#
# Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# This file is a part of the sglang-kunlun project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# Security Policy

## Supported Versions

Security fixes are applied to the latest release of `sglang-kunlun`. When a
security issue is confirmed, a patch is released for the current development
branch and backported to the latest stable release when applicable.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report the issue privately to the maintainers using one of the
following channels:

- Open a private security advisory on GitHub:
  <https://github.com/baidu-baige/sglang-kunlun/security/advisories/new>
- Contact the maintainers listed in [MAINTAINERS.md](MAINTAINERS.md).

Please include as much of the following as possible:

- A clear description of the vulnerability and its impact
- Steps to reproduce, or a minimal proof-of-concept
- The affected versions and runtime environment (SGLang, Kunlun runtime, OS)

We will acknowledge receipt of your report and keep you informed of the
investigation and remediation progress. We ask that you refrain from publicly
disclosing the issue until a fix has been released.

## Security Considerations for the Kunlun Runtime

`sglang-kunlun` is a runtime plugin: the Kunlun XPU runtime (`torch_xmlir`,
`kunlun_ops`, etc.) is provided by the deployment environment, not by this
repository. Ensure that the runtime, drivers, and Operator libraries installed
on the serving node are up-to-date with the latest security fixes.