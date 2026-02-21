# Security Review for FaceGallery

This document outlines the security posture of the FaceGallery project, detailing both implemented best practices and identified potential vulnerabilities with proposed mitigation strategies.

## Implemented Security Best Practices

FaceGallery incorporates several security-conscious design choices:

*   **Parameterized Queries**: All database interactions utilize parameterized queries, effectively preventing SQL Injection attacks by separating SQL code from user-supplied data.
*   **Secure Authentication**: The web application employs a session-based authentication mechanism. User PINs are stored as SHA-256 hashes, and session cookies are configured with the `HTTPOnly` flag to mitigate cross-site scripting (XSS) attacks from accessing session tokens.
*   **Robust Authorization**: Access control is enforced using Flask decorators (`@admin_required`, `@uploader_allowed`, `@login_required`), ensuring that users can only access resources and perform actions permitted by their assigned roles and permissions.
*   **Secure File Uploads**: When handling uploaded files, the `werkzeug.utils.secure_filename` utility is used to sanitize filenames, preventing path traversal attacks via maliciously crafted file names.
*   **Security Headers**: The web server configures `Cache-Control: no-store` headers for authenticated pages, preventing browsers from caching sensitive data and mitigating risks associated with users navigating back after logging out.

## Identified Vulnerabilities and Mitigation Strategies

The following potential vulnerabilities have been identified during a security review. We recommend addressing these to enhance the security of the FaceGallery application.

### 1. Informational: Default Administrator Credentials - **INTENTIONAL**

*   **Vulnerability Description**: Upon the first run of the web server (`run_web.py`), an administrator account named `admin` is automatically created with a default, easily guessable PIN of `1234` if no other users exist. This presents a informational vulnerability as it provides an immediate administrative backdoor if not promptly changed by the user.

### 2. Medium: Information Disclosure via Path Traversal (requires prior compromise) - **KNOWN**

*   **Vulnerability Description**: The web application serves images (`/photo/<id>`) and facilitates ZIP exports (`/export/zip`) based on file paths retrieved from the database. If an attacker could somehow manipulate these database entries (e.g., by exploiting the XSS vulnerability to gain admin privileges and then changing photo paths in the database), they might be able to craft requests to read arbitrary files from the server's file system (e.g., system configuration files like `/etc/passwd` or `boot.ini`). This vulnerability is conditional on a prior compromise allowing database manipulation.
*   **Location**: `./src/web/server.py` (`serve_photo`, `export_zip` routes)
*   **Mitigation Strategies**:
    *   **Strict Path Validation**: When serving or exporting files based on database paths, implement rigorous validation to ensure that the resolved file paths *must* reside within explicitly designated, authorized directories (e.g., `UPLOADS_ROOT_DIR` or configured scan folders). Prevent any path that attempts to traverse outside these boundaries.
    *   **Principle of Least Privilege**: Ensure the web server process runs with the absolute minimum necessary file system permissions. It should only have read/write access to its own application directories and the designated content folders, not the entire filesystem.
    *   **Prioritize XSS Fix**: Addressing the Stored XSS vulnerability (now fixed) significantly reduces the attack surface for this issue, as it removes a primary vector for manipulating database entries in a way that could lead to path traversal.

### 3. Risk Area: Insecure Dependencies - **KNOWN**

*   **Vulnerability Description**: FaceGallery relies on several complex third-party libraries for image processing and machine learning (e.g., Pillow, OpenCV, TensorFlow). These types of libraries, especially those that parse complex binary file formats like images, have historically been sources of severe vulnerabilities including remote code execution (RCE) and denial-of-service (DoS) attacks. While specified versions are relatively recent, these dependencies represent a significant and continuously evolving attack surface.
*   **Location**: `./requirements.txt`
*   **Mitigation Strategies**:
    *   **Regular Updates**: Establish a routine for regularly updating all project dependencies to their latest stable and secure versions.
    *   **Vulnerability Monitoring**: Subscribe to security advisories and vulnerability databases (e.g., NVD, dependabot alerts) for all critical dependencies.
    *   **Input Validation/Fuzzing**: Where feasible, implement additional custom input validation or fuzzing before feeding untrusted image data to these libraries to catch malformed inputs that might exploit parsing vulnerabilities.
    *   **Sandboxing**: For high-risk operations involving untrusted input processing, consider running the image processing components within isolated environments (e.g., Docker containers, separate microservices, or chroot jails) to limit the potential impact of an exploit.
