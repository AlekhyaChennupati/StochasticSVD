name := "Simple Project"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies ++=Seq("org.apache.spark" %% "spark-core" % "1.3.0",
"org.scalanlp" %% "breeze" % "0.10","org.scalanlp" %% "breeze-natives" % "0.10",
"org.apache.spark"  % "spark-mllib_2.10" % "1.3.0",
"org.apache.mahout" % "mahout-core" % "0.7"
)

resolvers ++= Seq("Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)
