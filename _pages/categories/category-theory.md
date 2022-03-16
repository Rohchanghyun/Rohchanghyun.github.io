---
title: "Theory"
layout: archive
permalink: categories/theory
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.theory | sort:"date" | reverse %}

{% for post in posts %} {% include archive-single2.html type=page.entries_layout %}{% endfor %}
