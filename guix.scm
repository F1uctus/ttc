;; To enter the Guix shell, run:
;;   $ guix shell -f guix.scm

(use-modules (guix gexp)
             (guix packages)
             (guix licenses)
             (guix git-download)
             (guix build-system python)
             (gnu packages check)
             (gnu packages python)
             (gnu packages python-xyz)
             (gnu packages python-build))


(define %source-dir (dirname (current-filename)))


(define ttc
  (package
  (name "ttc")
  (version "git")
  (source (local-file %source-dir
                      #:recursive? #t
                      #:select? (git-predicate %source-dir)))
  (build-system python-build-system)
  (propagated-inputs
   (list python-pypa-build
         python-tomli-w
         poetry))
  ;; TTC requires spacy <https://spacy.io/> but it is not packaged yet so it
  ;; will fail.
  (native-inputs
   (list python-poetry-core
         python-black
         python-numpy
         python-pytest))
  ;; (inputs
  ;;  (list python))
  (arguments
   (list
    #:tests? #t
    #:phases
    #~(modify-phases %standard-phases
        (delete 'sanity-check)
        (add-after 'unpack 'loosen-requirements
          (lambda _
            (substitute* "pyproject.toml"
              (("python = \">=3.8,<3.9\"")
               "python = \">=3.8,<=3.9.9\"")
              (("python_version = \"3.8\"")
               "python_version = \"3.9.9\""))
            (substitute* "poetry.lock"
              (("python-versions = \">=3.8,<3.9\"")
               "python-versions = \">=3.8,<=3.9.9\""))))
        (replace 'build
          (lambda _
            (invoke "python" "-m" "build" "--wheel" "--no-isolation" ".")))
        (replace 'check
          (lambda _
            (setenv "HOME" "/tmp")
            (invoke "poetry" "run" "pytest" "--maxfail=8")))
        (replace 'install
          (lambda _
            (let ((whl (car (find-files "dist" "\\.whl$"))))
              (invoke "pip" "--no-cache-dir" "--no-input"
                      "install" "--no-deps" "--prefix" #$output whl)))))))
  (home-page "https://github.com/F1uctus/ttc")
  (synopsis "Text-To-Conversation toolkit (TTC)")
  (description
   "Text-To-Conversation toolkit (TTC).")
  (license gpl3+)))

ttc
