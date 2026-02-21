# FaceGallery Security Declaration

As the author of FaceGallery, I understand that security and privacy are paramount, especially for a personal photo management application designed to keep your data local. This document serves as a transparent declaration of FaceGallery's security features, our design philosophy, and an honest acknowledgment of known security considerations.

## Our Core Security Philosophy

FaceGallery is built on the principle of **"Privacy First, Local Always."** This means:
*   **No Cloud Dependency**: Your photos and face data never leave your local machine. All processing, storage, and AI analysis happen locally.
*   **User Control**: You retain full control over your data and its access.
*   **Transparency**: We aim to be transparent about how data is handled and what security measures are in place.

## Key Security Features Implemented

We've integrated several best practices into FaceGallery to safeguard your data within its local operating environment:

*   **Robust Authentication and Authorization**:
    *   **PIN-based Access**: Web access is secured with individual user accounts and PINs.
    *   **Role-Based Access Control (RBAC)**: Users are assigned specific roles (`Admin`, `Uploader`, `Viewer`) with differing privileges.
    *   **Granular Permissions**: Viewers can be restricted to only see photos of specific individuals, ensuring tailored access.
    *   **HTTPOnly Session Cookies**: Session tokens are protected with the `HTTPOnly` flag, making them inaccessible to client-side scripts and mitigating many Cross-Site Scripting (XSS) risks.
*   **SQL Injection Protection**: All interactions with the SQLite database utilize parameterized queries, a fundamental defense that separates SQL code from user-supplied input, effectively preventing SQL Injection attacks.
*   **Secure File Handling and Storage**:
    *   **Confined Uploads**: All user-uploaded photos are strictly confined to a dedicated `UPLOADS_ROOT_DIR`.
    *   **Ownership Checks**: Folder management operations (add, remove, delete) on the web interface are tied to user ownership, preventing unauthorized users from managing folders they did not create or are not authorized for.
    *   **Filename Sanitization**: Uploaded filenames are sanitized using `werkzeug.utils.secure_filename` to prevent path traversal via malicious filenames.
    *   **Path Canonicalization**: File paths used internally are normalized and converted to absolute paths to prevent path traversal when accessing files.
*   **Data Protection Headers**: The web interface employs `Cache-Control: no-store` headers on authenticated pages to ensure sensitive data is not cached by browsers, reducing risks of information leakage, especially on shared machines.

## Acknowledged Security Considerations and Justifications

While we strive for robust security, it's important to acknowledge inherent challenges and design decisions.

### 1. Inadequate PIN Hashing (High Severity - Planned Improvement)

*   **Issue**: Currently, user PINs are hashed using SHA-256. While strong for general-purpose hashing, SHA-256 is not ideal for password storage as it lacks built-in salting and is computationally fast. This makes PINs vulnerable to offline brute-force attacks if the database is compromised.
*   **Justification/Context**: For a local-first application designed for ease of setup, a straightforward hashing approach was initially chosen.
*   **Future Plan**: We recognize this as a critical area for improvement. A migration to a stronger, purpose-built password hashing algorithm (e.g., `bcrypt`, `scrypt`, or `Argon2`) with proper per-user salting is a high-priority future enhancement.

### 2. Default Administrator Credentials (Informational - User Action Required)

*   **Issue**: Upon initial setup, an `admin` account with a default PIN (`1234`) is created for convenience if no other users exist. This presents an easily guessable initial credential.
*   **Justification/Context**: This is an intentional design choice for simplifying the initial user experience for a local, single-user focused application, ensuring immediate access to administrative functions.
*   **User Responsibility**: Users are strongly advised to change this default PIN immediately upon first login to a unique, strong PIN.

### 3. Information Disclosure via Path Traversal (Medium Severity - Mitigated by Design & User Context)

*   **Issue**: The web application serves files and creates ZIP exports based on file paths stored in the database. In theory, if an advanced attacker could *first* compromise the database (e.g., through a severe, undiscovered vulnerability) and *then* manipulate file paths, they might attempt to access arbitrary files on the system.
*   **Justification/Context**: FaceGallery implements several layers of defense. All file paths managed by the application are within defined, accessible directories (either user-configured scan folders or the `UPLOADS_ROOT_DIR`). Furthermore, `os.path.basename` is used for file-serving to prevent traversal within the served name. The primary protection comes from the application's local nature and the requirement of a prior, significant compromise of the database itself to exploit this. The web server also runs with minimal privileges where possible.
*   **Mitigation Efforts**: We continuously strive to ensure that all file system interactions are strictly validated against authorized base directories.

### 4. Reliance on Third-Party Dependencies (Risk Area - Ongoing Monitoring Required)

*   **Issue**: FaceGallery leverages powerful third-party libraries for image processing, AI, and web services (e.g., Pillow, OpenCV, Flask, InsightFace). These complex components, especially those handling binary file formats, can be sources of vulnerabilities (e.g., RCE, DoS) if not diligently maintained.
*   **Justification/Context**: Utilizing established libraries allows FaceGallery to deliver advanced features efficiently. This is a common and necessary practice in software development.
*   **Mitigation Efforts**: We commit to regular updates of all dependencies, monitoring security advisories (e.g., CVEs, dependabot), and adopting input validation/fuzzing where practical to reduce exposure to these inherent risks. For high-risk operations, running components in sandboxed environments is a long-term consideration.

## Our Commitment

We are committed to the ongoing security and privacy of FaceGallery. We encourage responsible disclosure of any discovered vulnerabilities.

**Author**: Dipta Roy
**Version**: 1.1.0
**Date**: 22 February 2026
