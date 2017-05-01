#!/usr/bin/env perl
# Generates Train/Test/Dev splits
# Every 8 lines is train, then one test, then one dev
# Usage: ./generate_splits.pl basename < input
# Output: basename.train, basename_test.gold, basename_dev.gold

use strict;
use Getopt::Long qw(:config bundling);

## Defaults
my $n = 8;
my ($print_stats, $min_length, $max_length, $train_stdout) = undef;
my %stats = ('total' => 0, 'train' => 0, 'dev' => 0, 'test' => 0, 'too_short' => 0, 'too_long' => 0);

my $usage = <<"END_OF_USAGE";
generate_splits.pl
Usage:    perl $0  basename < input
Function: Generates Train/Test/Dev splits
          Every n lines is train, then one test, then one dev
Options:
  -h, --help           Print usage
  -m, --min-length <n> Ignore lines containing fewer than <n> words
  -M, --max-length <n> Ignore lines containing more  than <n> words
  -n, --number <n>     Number of lines of training text (default: $n) for
                       every one line of dev and one line of test
  -s, --stats          Print some stats to STDERR upon completion
  -t, --train-stdout   Print training set to stdout
END_OF_USAGE

GetOptions(
    'h|help|?'         => sub { print $usage; exit; },
	'm|min-length=i'   => \$min_length,
	'M|max-length=i'   => \$max_length,
	'n|number=i'       => \$n,
	's|stats'          => \$print_stats,
	't|train-stdout'   => \$train_stdout,

) or die $usage;

my $basename = shift  or die "$0: Error - Specify file basename\n\n$usage";
if ($train_stdout) {
	open ( TRAIN, ">&", \*STDOUT ) or die;
} else {
	open ( TRAIN, ">", "${basename}.train" ) or die;
}
open ( TEST,  ">", "${basename}.test" ) or die;
open ( DEV,   ">", "${basename}.dev" ) or die;


my $counter;
while (<>) {
	$stats{'total'}++;
	my $length = 0;

	## Find out number of words in sentence, if we'll need it
	if ($min_length or $max_length) {
		my $trimmed = $_;
		$trimmed =~ s/^ +//g;
		$trimmed =~ s/ +$//g;
		$length = scalar(split /\s+/, $trimmed);
		#print "length: $length; $_";
	}

	## Input is either too short or too long; discard
	if ($min_length and $length < $min_length) {
		$stats{'too_short'}++;
		next;
	}
	elsif ($max_length and $length > $max_length) {
		$stats{'too_long'}++;
		next;
	}


    if ($counter == $n) {
		print TEST $_;
		$stats{'test'}++;
		$counter++;
    }
    elsif ($counter == $n + 1) {
		print DEV $_;
		$stats{'dev'}++;
		$counter = 0;
    }
    else {
		print TRAIN $_;
		$stats{'train'}++;
		$counter++;
    }
}

close TRAIN;
close TEST;
close DEV;

if ($print_stats) {
	print STDERR "Input     lines: $stats{'total'}\n";
	print STDERR "Too-short lines: $stats{'too_short'}\n";
	print STDERR "Too-long  lines: $stats{'too_long'}\n";
	print STDERR "Train     lines: $stats{'train'}\n";
	print STDERR "Dev       lines: $stats{'dev'}\n";
	print STDERR "Test      lines: $stats{'test'}\n";
}
