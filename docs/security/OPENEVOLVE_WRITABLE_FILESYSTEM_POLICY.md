# OpenEvolve writable-filesystem policy

The hardened candidate has one writable filesystem: private tmpfs `/workspace`. The concurrent entry limit counts every descendant—regular file, directory, link, FIFO, socket or device—and excludes only the mount root. Immutable image/input objects and trusted host evidence are outside this filesystem. The kernel inode ceiling is the declared limit plus one root inode; no unexplained margin is allowed.

Inode limits do not prevent multiple hard-linked names, so static policy rejects hard-link, symbolic-link, FIFO, mknod and Unix-socket-binding APIs. The supervisor independently walks the final workspace and accepts only regular files and directories. `/tmp`, `/var/tmp`, HOME and `/output` are not writable alternatives. Workspace bytes and individual file bytes have separate limits. Deletion releases concurrent quota; create/delete lifetime operations are not counted.
