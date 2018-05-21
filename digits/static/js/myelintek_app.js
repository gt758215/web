'use strict';

try {
(function() {
    var app = angular.module('mlt_app', [])

    app.controller('models_controller', function($scope, $controller) {
        $scope.title = 'Models';
        $scope.model_fields = [
            {name: 'Name', show: true, min_width: 100},
            {name: 'id', show: false, min_width: 200},
            {name: 'Status', show: true, min_width: 50},
            {name: 'Accuracy', show: true, min_width: 50},
            {name: 'Progress', show: true, min_width: 50},
            {name: 'Elapsed', show: true, min_width: 50},
            {name: 'Submitted', show: true, min_width: 50}
        ];

        $scope.print_time_diff_terse = function(diff) {
            return print_time_diff_terse(diff);
        };
    });

    app.controller("ds_controller", function($scope, $controller) {
        $scope.title = 'Datasets';
        $scope.fields = [{name: 'Name', show: true, min_width: 150},
                         {name: 'refs', show: false},
                         {name: 'extension', show: false, min_width: 150},
                         {name: 'Backend', show: true, min_width: 150},
                         {name: 'Status', show: true, min_width: 150},
                         {name: 'elapsed', show: false},
                         {name: 'Submitted', show: true, min_width: 150}];
    });

    app.controller("all_jobs_controller", function($scope, $http) {
        $scope.jobs = [];

        $scope.load_jobs = function() {
            $http({
                method: 'GET',
                url: '/completed_jobs.json',
            }).then(function success(response) {
                var r = response.data;
                $scope.jobs = [].concat(r.running, r.datasets, r.models);
            }, function error(response) {
                console.log(response.statusText);
            });
        };
        $scope.load_jobs();

        $scope.is_running = function(job) {
            return (job &&
                    (job.status == 'Initialized' ||
                     job.status == 'Waiting' ||
                     job.status == 'Running'));
        };

        $scope.not_running = function(job) {
            return (job &&
                    (job.status != 'Initialized' &&
                     job.status != 'Waiting' &&
                     job.status != 'Running'));
        };
        $scope.is_dataset = function(job) {
            return (job &&
                    (!$scope.is_running(job) &&
                     job.type == 'dataset'));
        };

        $scope.is_model = function(job) {
            return (job.type == 'model');
        };
    });

    app.directive('dgName', function() {
        return {
            restrict: 'AE',
            replace: true,
            template: (
                        '    <a href="' + URL_PREFIX + '/jobs/{[ job.id ]}" title="{[job.name]}">' +
                        '        {[ job.name ]}' +
                        '    </a>'),
        };
    });

    // Because jinja uses {{ and }}, tell angular to use {[ and ]}
    app.config(['$interpolateProvider', function($interpolateProvider) {
        $interpolateProvider.startSymbol('{[');
        $interpolateProvider.endSymbol(']}');
    }]);
})();
}
catch (ex) {
    console.log(ex);
}
