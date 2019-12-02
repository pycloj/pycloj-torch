(defproject pycloj/pycloj-torch "1.3.1-AUTO-0.1-SNAPSHOT"
  :description "pytorch wrapper for clojure"
  :url "https://github.com/pycloj/pycloj-torch"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.10.0"] [libpython-clj "1.6-SNAPSHOT"][alembic "0.3.2"]]
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
